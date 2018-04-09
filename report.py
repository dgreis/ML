import pandas as pd

from collections import defaultdict
from utils import *

class Report:

    def __init__(self,project_settings):
        self.project_settings = project_settings
        self.entries = dict()

    def add_entry(self,metric_name,model_name,content):
        entries = self.entries
        if entries.has_key(metric_name):
            entries[metric_name][model_name] = content
        else:
            entries[metric_name] = dict()
            entries[metric_name][model_name] = content
        self.entries = entries

    def write_report(self):
        project_settings = self.project_settings
        abs_project_dir = find_project_dir(project_settings)
        eval_dir = abs_project_dir + '/evaluation'
        if not os.path.isdir(eval_dir):
            os.makedirs(eval_dir)
        with open(eval_dir + '/report.html',"w") as f:
            eval_pak = fetch_eval_pak(project_settings)
            entries = self.entries
            column_entries = {k: v for k, v in entries.iteritems() if eval_pak['battery'][k]['metric_type'] == 'column'}
            column_table = pd.DataFrame(column_entries)
            column_table_html = column_table.T.to_html()
            f.write('<h1>Overall</h1>')
            f.write(column_table_html)
            array_entries = {k: v for k, v in entries.iteritems() if eval_pak['battery'][k]['metric_type'] == 'array'}
            for entry in array_entries:
                f.write("<h1>" + entry + "</h1>")
                models = array_entries[entry]
                for name in models:
                    f.write("<h3>" + name + "</h3>")
                    array = models[name]
                    array_table = pd.DataFrame(array)
                    array_table_html = array_table.to_html()
                    f.write(array_table_html)
            cr_entries = {k: v for k, v in entries.iteritems() if eval_pak['battery'][k]['metric_type'] == 'classification_report'}
            for entry in cr_entries:
                f.write("<h1>" + entry + "</h1>")
                models = cr_entries[entry]
                for name in models:
                    f.write("<h3>" + name + "</h3>")
                    element_table = self.create_cr_table(models[name])
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

