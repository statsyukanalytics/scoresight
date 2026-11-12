import unittest
from unittest.mock import Mock, patch
import numpy as np
from studio.scoreboard.field_layouts import profile_layouts, alternatives_for, merge_readings, EMPTY, CLOCK_CHARACTERS, PERIOD_CHARACTERS
from studio.scoreboard.auto_fields import AutoReader
from studio.scoreboard.runtime import batch_frames

class LayoutRecoveryTests(unittest.TestCase):
    def test_alternate_layout_survives_missing_primary_but_conflicts_are_empty(self):
        layouts=[[(0,0,10,10),(10,0,10,10)],[(20,0,10,10),(30,0,10,10)]]
        good=[('10:00',.98),('1ST',.98)]
        self.assertEqual(merge_readings(layouts,[('',0),('',0)]+good),(tuple(good),layouts[1]))
        self.assertEqual(merge_readings(layouts,good+[('09:00',.99),('1ST',.99)]),(EMPTY,None))
        self.assertEqual(merge_readings(layouts,good+good)[0],tuple(good))

    def test_profiles_need_detected_geometry_before_variants_are_used(self):
        config={'width':1280,'height':720,'clock':[128,87,92,31],'period':[228,87,52,31],
                'variants':[{'clock':[48,87,100,31],'period':[150,87,48,31]}]}
        with patch('studio.scoreboard.field_layouts.profiles',return_value=[config]):
            layouts=alternatives_for(profile_layouts(config)[0],1280,720)
            self.assertEqual(len(layouts),2)
            unrelated=[(800,400,80,30),(900,400,50,30)]
            self.assertEqual(alternatives_for(unrelated,1280,720),[unrelated])
            scaled=[(64,44,46,16),(114,44,26,16)]
            self.assertGreater(len(alternatives_for(scaled,640,360)),1)

    def test_manual_variants_are_validated_and_all_retained(self):
        config={'width':100,'height':50,'clock':[0,0,20,10],'period':[20,0,10,10],
                'variants':[{'clock':[40,0,20,10],'period':[60,0,10,10]}]}
        self.assertEqual(len(profile_layouts(config)),2)
        config['variants'][0]['clock']=[99,0,20,10]
        with self.assertRaises(ValueError):profile_layouts(config)

    def test_known_layouts_decode_by_role_and_move_without_waiting_for_search(self):
        rec=Mock();reader=AutoReader(Mock(),rec)
        first=[(0,0,5,5),(5,0,5,5)];second=[(10,0,5,5),(15,0,5,5)]
        reader.fields=first;reader.layouts=[first,second];reader.until_search=10
        rec.recognize.return_value=[('',0),('',0),('10:00',.99),('1ST',.99)]
        reader._search=Mock()
        result=reader.read(np.zeros((20,20,3),dtype=np.uint8))
        self.assertEqual(result[1],second)
        self.assertEqual(rec.recognize.call_args.args[1],[CLOCK_CHARACTERS,PERIOD_CHARACTERS]*2)
        reader._search.assert_not_called()

    def test_cpu_and_cuda_batch_limits_are_independent_and_bounded(self):
        with patch.dict('os.environ',{'SCORESIGHT_DEVICE':'cpu'},clear=True):self.assertEqual(batch_frames(),8)
        with patch.dict('os.environ',{'SCORESIGHT_DEVICE':'cuda'},clear=True):self.assertEqual(batch_frames(),16)
        with patch.dict('os.environ',{'SCORESIGHT_BATCH_FRAMES':'0'},clear=True):
            with self.assertRaises(ValueError):batch_frames()

if __name__=='__main__':unittest.main()
