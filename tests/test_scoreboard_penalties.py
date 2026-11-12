import unittest

from studio.scoreboard.penalty_schedule import derive_windows



def event(t,kind,team=None,code=None,number=1):
    period=min(3,t//1200+1);elapsed=t-(period-1)*1200
    details={'eventOwnerTeamId':team}
    if kind=='penalty':details.update(typeCode='MIN',duration=2)
    return dict(event_id=number,sort_order=number,period_number=period,period_type='REG',
                time_in_period=f'{elapsed//60:02d}:{elapsed%60:02d}',
                type_desc_key=kind,situation_code=code,details=details)


def snapshot(events):
    entries=events+[event(3600,'period-end',number=len(events)+1)]
    for i,e in enumerate(entries):e['sort_order']=i
    return dict(game=dict(awayTeam_id=4,homeTeam_id=13,awayTeam_abbrev='PHI',homeTeam_abbrev='FLA'),events=entries)


def penalty(t, team, number, kind='MIN', duration=2):
    e = event(t, 'penalty', team, number=number)
    e['details'].update(typeCode=kind, duration=duration)
    return e


class ExtendedPenaltyChecks(unittest.TestCase):
    def test_major_survives_powerplay_goal(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1, 'MAJ', 5),
                                           event(40, 'goal', 13, '1451', 2)]))
        self.assertEqual(windows[0]['end_game_seconds'], 310)

    def test_double_minor_goal_starts_second_half(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1, duration=4),
                                           event(40, 'goal', 13, '1451', 2)]))
        self.assertEqual(windows[0]['end_game_seconds'], 160)

    def test_double_minor_second_half_goal_ends_advantage(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1, duration=4),
                                           event(150, 'goal', 13, '1451', 2)]))
        self.assertEqual(windows[0]['end_game_seconds'], 150)

    def test_matching_majors_cancel_but_extra_minor_remains(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1, 'MAJ', 5),
                                           penalty(10, 13, 2, 'MAJ', 5), penalty(10, 4, 3)]))
        self.assertEqual(windows[0]['penalty_event_ids'], [3])
        self.assertEqual(windows[0]['end_game_seconds'], 130)

    def test_one_minor_each_is_four_on_four_then_extra_is_four_on_three(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1), penalty(10, 13, 2),
                                           event(11, 'faceoff', 4, '1441', 3), penalty(30, 4, 4),
                                           event(50, 'goal', 13, '1341', 5),
                                           event(51, 'faceoff', 4, '1441', 6)]))
        self.assertEqual(len(windows), 1)
        self.assertEqual((windows[0]['start_game_seconds'], windows[0]['end_game_seconds']), (30, 50))
        self.assertEqual(windows[0]['phases'][0]['away_skaters'], 3)

    def test_multiple_coincidental_minors_do_not_reduce_manpower(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1), penalty(10, 13, 2),
                                           penalty(10, 4, 3), penalty(10, 13, 4),
                                           event(11, 'faceoff', 4, '1551', 5)]))
        self.assertEqual(windows, [])

    def test_coincidental_pair_during_existing_penalty_is_substituted(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1), penalty(30, 4, 2), penalty(30, 13, 3),
                                           event(31, 'faceoff', 4, '1451', 4)]))
        self.assertEqual(windows[0]['penalty_event_ids'], [1])

    def test_bench_minor_and_game_misconduct(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1, 'BEN'), penalty(10, 4, 2, 'GAM', 10)]))
        self.assertEqual(windows[0]['end_game_seconds'], 130)

    def test_expiry_event_may_record_instant_before_return(self):
        windows = derive_windows(snapshot([penalty(10, 4, 1), event(130, 'shot-on-goal', 13, '1451', 2)]))
        self.assertEqual(windows[0]['end_game_seconds'], 130)
        with self.assertRaisesRegex(ValueError, 'disagrees'):
            derive_windows(snapshot([penalty(10, 4, 1), event(131, 'shot-on-goal', 13, '1451', 2)]))

    def test_overtime_without_penalties_preserves_regulation_windows(self):
        data = snapshot([penalty(10, 4, 1)])
        ot = event(3601, 'goal', 4, '1331', 3)
        ot.update(period_number=4, period_type='OT', time_in_period='00:01')
        data['events'].append(ot)
        self.assertEqual(len(derive_windows(data)), 1)


if __name__ == '__main__':
    unittest.main()
