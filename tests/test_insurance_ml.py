import tempfile
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from insurance_ml_phoenix import read_arff,load_policies,metrics,FEATURES

class InsuranceEvaluationTests(unittest.TestCase):
    def test_errors_have_known_values(self):
        m=metrics([0,100],[10,80],[.5,1])
        self.assertEqual(m['mae_eur'],15)
        self.assertAlmostEqual(m['rmse_eur'],np.sqrt(250))
        self.assertEqual(m['mean_underprediction_eur'],5)
        self.assertAlmostEqual(m['predicted_to_actual_ratio'],.9)

    def test_no_claim_or_identifier_is_a_feature(self):
        self.assertTrue({'ClaimNb','ClaimAmount','IDpol','Exposure'}.isdisjoint(FEATURES))

    def test_inflation_changes_outcomes_without_predictions(self):
        predictions=np.array([10.,80.]);before=predictions.copy()
        a=metrics([0,100],predictions,[.5,1])
        b=metrics(np.array([0,100])*1.25,predictions,[.5,1])
        np.testing.assert_array_equal(predictions,before)
        self.assertEqual(b['mean_actual_eur'],a['mean_actual_eur']*1.25)
        self.assertEqual(b['mae_eur'],27.5)

    def test_multiple_claims_aggregate_before_join(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            columns=['IDpol','ClaimNb','Exposure']+FEATURES
            header='@relation test\n'+''.join('@attribute '+c+' numeric\n' for c in columns)+'@data\n'
            rows=[]
            for i,n in [(1,2),(2,0),(3,1)]:
                rows.append(','.join(map(str,[i,n,1,'A',5,2,40,50,'B1','Diesel',100,'R11'])))
            (folder/'freMTPL2freq.arff').write_text(header+'\n'.join(rows))
            (folder/'freMTPL2sev.arff').write_text('@relation test\n@attribute IDpol numeric\n@attribute ClaimAmount numeric\n@data\n1,20\n1,30\n99,10\n')
            df,audit=load_policies(folder)
            self.assertEqual(len(df),3)
            self.assertEqual(df.set_index('IDpol').loc[1,'ClaimAmount'],50)
            self.assertEqual(df.set_index('IDpol').loc[2,'ClaimAmount'],0)
            self.assertEqual(audit['unmatched_severity_policy_ids'],1)
            self.assertEqual(audit['severity_rows_beyond_first_claim'],1)
            self.assertEqual(audit['policies_with_positive_claim_count_but_no_positive_amount'],1)

if __name__=='__main__':unittest.main()
