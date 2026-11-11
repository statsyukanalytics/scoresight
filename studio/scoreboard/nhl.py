"""Read one game's timestamped records from the existing CapAngry database."""
import os


def load_game(game_id):
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
    if not events:
        raise ValueError(f'No NHL events found for game {game_id}')
    return events, shifts
