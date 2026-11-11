"""Read one game's timestamped records from the existing CapAngry database."""
import os


def load_bundle(game_id):
    import psycopg
    from psycopg.rows import dict_row

    url = os.environ.get('NHL_DATABASE_URL')
    if not url:
        raise ValueError('Set NHL_DATABASE_URL to the CapAngry NHL database connection string')
    with psycopg.connect(url, row_factory=dict_row, connect_timeout=15) as conn:
        conn.execute('SET TRANSACTION READ ONLY')
        conn.execute("SET LOCAL statement_timeout = '30s'")
        # Keep the complete event payload: scorer, assists, coordinates, penalties, etc.
        events = conn.execute('SELECT * FROM expanded_play_by_play WHERE game_id = %s ORDER BY sort_order, event_id', (game_id,)).fetchall()
        shifts = conn.execute('SELECT * FROM player_shifts WHERE game_id = %s ORDER BY period, start_time, player_id', (game_id,)).fetchall()
        game = conn.execute('SELECT * FROM games WHERE id = %s', (game_id,)).fetchone()
        player_ids = sorted({s['player_id'] for s in shifts} | {v for e in events for k, v in e.items() if k.endswith('PlayerId') and isinstance(v, int)})
        team_ids = [game['homeTeam_id'], game['awayTeam_id']] if game else []
        season = game['season'] if game else None
        queries = {
            'raw_events': ('SELECT * FROM play_by_play WHERE game_id = %s ORDER BY sort_order, event_id', (game_id,)),
            'shot_context': ('SELECT * FROM xg_table WHERE game_id = %s', (game_id,)),
            'goal_visualizer': ('SELECT * FROM goal_visualizer WHERE game_id = %s', (game_id,)),
            'players': ('SELECT * FROM general_player_data WHERE "playerId" = ANY(%s)', (player_ids,)),
            'teams': ('SELECT * FROM teams WHERE nhl_team_id = ANY(%s)', (team_ids,)),
            'rosters': ('SELECT * FROM team_rosters WHERE season = %s AND nhl_team_id = ANY(%s)', (season, team_ids)),
            'player_game_logs': ('SELECT * FROM player_game_logs WHERE "gameId" = %s', (game_id,)),
            'goalie_game_logs': ('SELECT * FROM goalie_game_logs WHERE "gameId" = %s', (game_id,)),
            'player_game_xg': ('SELECT * FROM xg_by_game WHERE "gameId" = %s', (game_id,)),
            'skater_season_stats': ('SELECT * FROM skater_stats WHERE season = %s AND "playerId" = ANY(%s)', (season, player_ids)),
            'goalie_season_stats': ('SELECT * FROM goalie_stats WHERE season = %s AND "playerId" = ANY(%s)', (season, player_ids)),
            'draft_picks': ('SELECT * FROM draft_picks WHERE player_id = ANY(%s)', (player_ids,)),
            'contracts': ('SELECT * FROM salary_data WHERE "PlayerID" = ANY(%s)', (player_ids,)),
        }
        context = {'game': game}
        unavailable = {}
        for key, (query, params) in queries.items():
            try:
                with conn.transaction():
                    context[key] = conn.execute(query, params).fetchall()
            except psycopg.Error as exc:
                # Enrichment availability must not discard the canonical timeline.
                unavailable[key] = exc.sqlstate or type(exc).__name__
    if not events:
        raise ValueError(f'No NHL events found for game {game_id}')
    return {'events': events, 'shifts': shifts, 'context': context, 'unavailable_context': unavailable}


def load_game(game_id):
    bundle = load_bundle(game_id)
    return bundle['events'], bundle['shifts']
