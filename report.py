import pandas as pd

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
        eval_dir = project_settings['repo_loc'] + '/' + project_settings['project_name'] + '/evaluation'
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
            text_entries = {k: v for k, v in entries.iteritems() if eval_pak['battery'][k]['metric_type'] == 'text'}
            for entry in text_entries:
                f.write("<h1>" + entry + "</h1>")
                models = text_entries[entry]
                for name in models:
                    f.write("<h3>" + name + "</h3>")
                    element_text = models[name]
                    element_html = "<p>" + element_text + "</p>"
                    f.write(element_html)



