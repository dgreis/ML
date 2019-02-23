import pandas as pd

from collections import defaultdict
from utils import update, find_project_dir
import os
from evaluation import load_evaluation_battery

class Report:

    def __init__(self,project_settings):
        self.project_settings = project_settings
        self.report_entries = pd.DataFrame()

    def add_entries_to_report(self,entries):
        entry_columns = ['dataset_name','model_name','setting', 'metric','content']
        if type(entries) == pd.DataFrame:
            entries = [entries.iloc[i].to_dict() for i in range(len(entries))]
            for entry in entries:
                assert set(entry_columns) == set(entry.keys())
        else:
            assert set(entry_columns) == set(entries.keys())
        report_entries = self.report_entries
        new_report_entries = report_entries.append(entries,ignore_index=True)
        assert len(new_report_entries) > len(report_entries)
        self.report_entries = new_report_entries

    def gen_column_table(self, report_entries, dataset_name):
        project_settings = self.project_settings
        eval_battery = load_evaluation_battery(project_settings)
        mtlu = dict([(b, eval_battery[b]['metric_type']) for b in eval_battery.keys()])
        report_entries['metric_type'] = report_entries['metric'].apply(lambda x: mtlu[x])
        column_entries = report_entries[(report_entries['dataset_name'] == dataset_name) &
                                        (report_entries['metric_type'] == 'column')]
        column_table = column_entries[['metric', 'model_name', 'content']].pivot('metric', 'model_name', 'content')
        column_table = column_table.rename_axis(None)
        for col in column_table.columns:
            column_table[col] = column_table[col].apply(lambda x: "%0.3f (+/-%0.03f)" % (x[0], x[1] * 2) #TODO: Why *2?
                                if type(x) == tuple
                                else "{0:1.3f}".format(x))
        return column_table

    def write_report(self): #TODO: Adapt this for classification val/test output. Now only suited to regression/col metrics
        project_settings = self.project_settings
        report_entries = self.report_entries
        abs_project_dir = find_project_dir(project_settings)
        eval_dir = abs_project_dir + '/evaluation'
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        with open(eval_dir + '/report.html',"w") as f:
            cv_column_table = self.gen_column_table(report_entries, 'cross_validation')
            if len(cv_column_table) > 0:
                cv_column_table_html = cv_column_table.to_html()
                f.write('<h1>Cross-Validated Metrics</h1>')
                f.write(cv_column_table_html)
            val_column_table = self.gen_column_table(report_entries, 'validation')
            val_column_table_html = val_column_table.to_html()
            f.write('<h1>Validation Set Metrics</h1>')
            f.write(val_column_table_html)
            if 'test' in report_entries['dataset_name'].values:
                test_column_table = self.gen_column_table(report_entries, 'test')
                test_column_table_html = test_column_table.to_html()
                f.write('<h1>Test Set Metrics</h1>')
                f.write(test_column_table_html)
            array_entries = report_entries[(report_entries['dataset_name'] == 'validation') &
                                            (report_entries['metric_type'] == 'array')].reset_index(drop=True)
            array_metric_names = array_entries['metric'].unique().tolist()
            for array_metric_name in array_metric_names:
                f.write("<h1>" + array_metric_name + "</h1>")
                model_rows = array_entries[array_entries['metric'] == array_metric_name].reset_index()
                for i in range(len(model_rows)):
                    row = dict(model_rows.iloc[i])
                    f.write("<h3>" + row['model_name'] + "</h3>")
                    array = row['content']
                    array_table = pd.DataFrame(array)
                    array_table_html = array_table.to_html()
                    f.write(array_table_html)
            cr_entries = report_entries[(report_entries['dataset_name'] == 'validation') & (report_entries['metric_type'] == 'classification_report')].reset_index(drop=True)
            if len(cr_entries) > 0:
                f.write("<h1>" + 'classification_report' + "</h1>")
            for i in range(len(cr_entries)):
                row = dict(cr_entries.iloc[i])
                f.write("<h3>" + row['model_name'] + "</h3>")
                element_table = self.create_cr_table(row['content'])
                element_html = element_table.to_html()
                f.write(element_html)

    def create_cr_table(self,blob):
        # Parse rows
        tmp = list()
        for row in blob.split("\n"):
            parsed_row = [x for x in row.split("  ") if len(x) > 0]
            if len(parsed_row) > 0:
                tmp.append(parsed_row)

        # Store in dictionary
        measures = tmp[0]

        D_class_data = defaultdict(dict)
        for row in tmp[1:]:
            class_label = row[0]
            for j, m in enumerate(measures):
                D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
        return pd.DataFrame(D_class_data).T

