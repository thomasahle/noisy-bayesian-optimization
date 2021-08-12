import asyncio
import itertools
import random
import logging

import chess.pgn
import chess.engine
import chess
from chess import WHITE, BLACK


class Arena:
    MATE_SCORE = 32767

    def __init__(self, enginea, engineb, book, limit, max_len, win_adj_count, win_adj_score):
        self.enginea = enginea
        self.engineb = engineb
        self.book = book
        self.limit = limit
        self.max_len = max_len
        self.win_adj_count = win_adj_count
        self.win_adj_score = win_adj_score

    def adjudicate(self, score_hist):
        if len(score_hist) > self.max_len:
            return '1/2-1/2'
        # Note win_adj_count is in moves, not plies
        count_max = self.win_adj_count * 2
        if count_max > len(score_hist):
            return None
        # Test if white has been winning. Notice score_hist is from whites pov.
        if all(v >= self.win_adj_score for v in score_hist[-count_max:]):
            return '1-0'
        # Test if black has been winning
        if all(v <= -self.win_adj_score for v in score_hist[-count_max:]):
            return '0-1'
        return None

    async def play_game(self, init_node, game_id, white, black):
        """ Yields (play, error) tuples. Also updates the game with headers and moves. """
        try:
            game = init_node.root()
            node = init_node
            score_hist = []
            for ply in itertools.count(int(node.board().turn == BLACK)):
                board = node.board()

                adj_result = self.adjudicate(score_hist)
                if adj_result is not None:
                    game.headers.update({
                        'Result': adj_result,
                        'Termination': 'adjudication'
                    })
                    return

                if board.is_game_over(claim_draw=True):
                    result = board.result(claim_draw=True)
                    game.headers["Result"] = result
                    return

                # Try to actually make a move
                play = await (white, black)[ply % 2].play(
                    board, self.limit, game=game_id,
                    info=chess.engine.INFO_BASIC | chess.engine.INFO_SCORE)
                yield play, None

                if play.resigned:
                    game.headers.update({'Result': ('0-1', '1-0')[ply % 2]})
                    node.comment += f'; {("White","Black")[ply%2]} resgined'
                    return

                node = node.add_variation(play.move, comment=
                        f'{play.info.get("score",0)}/{play.info.get("depth",0)}'
                        f' {play.info.get("time",0)}s')

                # Adjudicate game by score, save score in wpov
                try:
                    score_hist.append(play.info['score'].white().score(
                        mate_score=max(self.win_adj_score, Arena.MATE_SCORE)))
                except KeyError:
                    logging.debug('Engine didn\'t return a score for adjudication. Assuming 0.')
                    score_hist.append(0)

        except (asyncio.CancelledError, KeyboardInterrupt) as e:
            print('play_game Cancelled')
            # We should get CancelledError when the user pressed Ctrl+C
            game.headers.update({'Result': '*', 'Termination': 'unterminated'})
            node.comment += '; Game was cancelled'
            await asyncio.wait([white.quit(), black.quit()])
            yield None, e
        except chess.engine.EngineError as e:
            game.headers.update(
                {'Result': ('0-1', '1-0')[ply % 2], 'Termination': 'error'})
            node.comment += f'; {("White","Black")[ply%2]} terminated: {e}'
            yield None, e

    async def configure(self, args):
        # We configure enginea, engineb is our unchanged opponent.
        # Maybe this should be refactored.
        self.enginea.id['args'] = args
        self.engineb.id['args'] = {}
        try:
            await self.enginea.configure(args)
        except chess.engine.EngineError as e:
            print(f'Unable to start game {e}')
            return [], 0

    async def run_games(self, game_id=0, games_played=2):
        score = 0
        games = []
        assert games_played % 2 == 0
        for r in range(games_played//2):
            init_board = random.choice(self.book)
            for color in [WHITE, BLACK]:
                white, black = (self.enginea, self.engineb) if color == WHITE \
                    else (self.engineb, self.enginea)
                game_round = games_played * game_id + color + 2*r
                game = chess.pgn.Game({
                    'Event': 'Tune.py',
                    'White': white.id['name'],
                    'WhiteArgs': repr(white.id['args']),
                    'Black': black.id['name'],
                    'BlackArgs': repr(black.id['args']),
                    'Round': game_round
                })
                games.append(game)
                # Add book moves
                game.setup(init_board.root())
                node = game
                for move in init_board.move_stack:
                    node = node.add_variation(move, comment='book')
                # Run engines
                async for _play, er in self.play_game(node, game_round, white, black):
                    # If an error occoured, return as much as we got
                    if er is not None:
                        return games, score, er
                result = game.headers["Result"]
                if result == '1-0' and color == WHITE or result == '0-1' and color == BLACK:
                    score += 1
                if result == '1-0' and color == BLACK or result == '0-1' and color == WHITE:
                    score -= 1
        return games, score/games_played, None


class ArenaRunner:
    def __init__(self, engines, opt, x_to_args, n_games, concurrency=1, games_per_encounter=1):
        self.engines = engines
        self.opt = opt
        self.concurrency = concurrency
        assert len(engines) == concurrency
        self.started = 0
        self.games_per_encounter = games_per_encounter
        self.x_to_args = x_to_args
        self.n_games = n_games

    def _on_done(self, task):
        if task.exception():
            logging.error('Error while excecuting game')
            task.print_stack()

    def _new_game(self, arena):
        async def routine(game_id):
            x = await self.opt.ask()
            engine_args = self.x_to_args(x)
            print(f'Starting {self.games_per_encounter} games {game_id}/{self.n_games} with {engine_args}')
            await arena.configure(engine_args)
            res = await arena.run_games(game_id=game_id, games_played=self.games_per_encounter)
            return x, res
        task = asyncio.create_task(routine(self.started))
        # We tag the task with some attributes that we need when it finishes.
        setattr(task, 'tune_arena', arena)
        setattr(task, 'tune_game_id', self.started)
        task.add_done_callback(self._on_done)
        self.started += 1
        return task

    async def run(self, arena_args):
        # Find how many games are already in the optimizer
        self.started = await self.opt.size()

        tasks = []
        if self.n_games - self.started > 0:
            xs = await self.opt.ask(min(self.concurrency, self.n_games - self.started))
        else:
            xs = []
        assert len(xs) <= self.concurrency

        for conc_id, x_init in enumerate(xs):
            enginea, engineb = self.engines[conc_id]
            arena = Arena(enginea, engineb, *arena_args)
            tasks.append(self._new_game(arena))

        while tasks:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            tasks = list(pending)
            for task in done:
                arena, game_id = task.tune_arena, task.tune_game_id
                x, (games, y, er) = task.result()

                yield (game_id, x, y, games, er)

                await self.opt.tell(x, (y+1)/2) # Format in [0,1]

                if self.started < self.n_games:
                    tasks.append(self._new_game(arena))

