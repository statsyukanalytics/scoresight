"""Decode NHL event situation codes; all matchups are away skaters vs home skaters."""
import re


def decode_strength(code):
    state = dict(situation_code=code, classification='unknown', matchup=None,
                 away_skaters=None, home_skaters=None, away_goalie_present=None,
                 home_goalie_present=None, away_state='unknown', home_state='unknown',
                 basis='NHL recorded situation at event; not a continuous clip label')
    if not isinstance(code, str) or re.fullmatch(r'[01][3-6][3-6][01]',code) is None:
        return state
    away_goalie, away, home, home_goalie = map(int,code)
    state.update(away_skaters=away,home_skaters=home,away_goalie_present=bool(away_goalie),
                 home_goalie_present=bool(home_goalie),matchup=f'{away}v{home}')
    if not away_goalie or not home_goalie:
        state.update(classification='goalie_pulled',
                     away_state='goalie_pulled' if not away_goalie else 'opponent_goalie_pulled',
                     home_state='goalie_pulled' if not home_goalie else 'opponent_goalie_pulled')
    elif max(away,home)<=5:
        if away==home:
            state.update(classification='even_strength',away_state='EV',home_state='EV')
        else:
            state.update(classification='power_play',away_state='PP' if away>home else 'PK',
                         home_state='PP' if home>away else 'PK')
    return state
