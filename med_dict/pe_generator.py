# -*- coding: utf-8 -*-

import pandas as pd

def generator():
#    pe_names_eng = ['Triglycerides', 'BloodPressure', 'FastingGlucose', 'Cholesterol', 'DistantVision(Uncorrected)', 'PULSE', 'BodyHeight', 'FluVaccination']
    col_names = ['Testitems', 'item', 'items', '测试项目']
    df = pd.DataFrame({'col_names':col_names})
    df.to_csv('physical_exam.csv', index=False)

if __name__ == '__main__':
    print("starting generate physical exam dictionary.")
    df = generator()
    