import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def clean_date(date_obj) -> object:
        """Cleans str object in date format into datetime object. 
        Returns NaT for missing values and -1 for invalid formats
        """

        if type(date_obj) == float:
            return pd.NaT
        #Following US Date Format Month-Day-Year
        try:
            #Formatted Month Day, Year -> Year, Month, Day
            date = datetime.strptime(date_obj, '%B %d, %Y').date()
            return date
        except:
            pass
        try:
            #Formatted Month Year -> Year, Month, Day(1st)
            date = datetime.strptime(date_obj, '%B %Y').date()
            return date
        except:
            #Failed both formats
            print('Failed to convert date object:', date_obj)
            return -1
def gen_future_flag(df, date_col) -> pd.Series:
    """Creates flag column of future dates"""
    today_date = datetime.today().date()
    future_flag = str(date_col) + " Future Flag"
    df[future_flag] = np.where((df[date_col] > today_date), 1, 0)    
    return

def gen_out_of_order_flag(df, date_cols) -> pd.Series:
    """Checks if two columns of dates are in the right order. 
    Columns from Older to Newer dates"""
    flag_var = str(date_cols[0]) + " & " + str(date_cols[1]) + " Invalid Order Flag"
    df[flag_var] = np.where((df[date_cols[1]] > df[date_cols[0]]), 0, 1)
    return

def gen_date_report_df(df, date_columns) -> pd.DataFrame:
    """Creates a final missing data report on all date columns. 
    Includes percent missing for visualizations"""
    def gen_date_report(df, date_col):
        """Helper function to create dataframe
        """
        future_flag = str(date_col) + " Future Flag"
        null_counts = sum(df[date_col].isna())
        invalid_dates = len(df[df[date_col] == -1][date_col]) 
        future_dates = sum(df[future_flag])
        valid_dates = len(df[date_col]) - \
                        (null_counts + invalid_dates + future_dates)
        report = {"Total Count": len(df[date_col]), 
                "Null Count": null_counts, 
                "Invalid Date Count": invalid_dates, 
                "Future Date Count" : future_dates,
                "Current Date Count": valid_dates}
        return report

    report_df = pd.DataFrame()
    for date_col in date_columns:
        col_df = pd.DataFrame([{'Column': date_col}])
        val_df = pd.DataFrame([gen_date_report(df, date_col)])
        temp_df = pd.concat([col_df, val_df], axis = 1)
        report_df = pd.concat([report_df, temp_df])
    
    report_df['Null Percent'] = round(((report_df['Null Count'] / 
                                        report_df['Total Count']) * 100), 2)
    report_df['Invalid Date Percent'] = round(((report_df['Invalid Date Count'] / 
                                        report_df['Total Count']) * 100), 2)
    report_df['Future Date Percent'] = round(((report_df['Future Date Count'] / 
                                        report_df['Total Count']) * 100), 2)

    report_df = report_df.reset_index(drop=True)
    return report_df

def gen_full_date_report(df, keep_cols) -> pd.DataFrame:
    """Generates columns specific to date columns"""
    #df2 = df[df.columns.difference(keep_cols)]
    df2 = df[keep_cols]
    return df2

def gen_date_plot(df) -> None:
    """Plots report metrics for specific column"""
    col_name = df['Column']
    x = list(df.index[1:5])
    x_range = np.arange(0, len(x), 1, dtype = int)
    y = list(df.values[1:5])
    z = list(df.values[6:])

    my_colors = ['dodgerblue', 'lightcoral', 'darkred', 'slategray']
    plt.rcParams['figure.figsize'] = [15, 5]
    plt.bar(x,y, color = my_colors)
    plt.xticks(x, fontsize = 20)
    plt.yticks(fontsize = 20)

    for i, v in enumerate(y):
        if i == 0:
            plt.text(x_range[i] + 0.01, (y[i]+10), str(v), 
                    horizontalalignment = 'center', fontsize = 20, 
                    bbox=dict(facecolor='white', alpha=0.5))
        else:
            plt.text(x_range[i] + 0.01, (y[i]+10), str(v) + '  (' + str(z[i-1]) + '%)', 
                    horizontalalignment = 'center', fontsize = 20, 
                    bbox=dict(facecolor='white', alpha=0.5))

    plt.xlabel(str(col_name), fontsize = 20)
    plt.ylabel("Counts", fontsize = 20)
    plt.title(str(col_name) + ' Report', fontsize = 20)
    plt.show()
    return

def gen_date_overview_plot(df) -> None:
    """Plots report metrics for all date columns"""    
    my_colors = ['dodgerblue', 'skyblue', 'lightcoral', 'darkred', 'slategray']
    plot_df = df[['Column','Total Count', 'Current Date Count', 'Null Count', 'Invalid Date Count', 'Future Date Count']]
    
    plt.rcParams['figure.figsize'] = [25, 10]
    ax = plot_df.plot(kind = 'barh',width = 0.8, color = my_colors)
    x_labels = list(plot_df['Column'])
    x_range = np.arange(0, len(list(plot_df['Column'])), 1, dtype = int)
    for c in ax.containers:
        ax.bar_label(c, label_type='edge')

    plt.xticks(fontsize = 20)
    plt.yticks(x_range, x_labels,fontsize = 20)
    plt.xlabel('Counts', fontsize = 20)
    plt.title('All Date Columns Report', fontsize = 20)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = [4,3,2,1,0]
    plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], 
                loc = 'upper right', bbox_to_anchor=(1.09, .905))
    plt.show()
    return

def clean_age(age_obj):

    age_obj = str(age_obj).lower()
    age_list = age_obj.split( '\xa0')
    age_list = [x.strip() for x in age_list]
    if len(age_list) == 0:
        #print("Returned default")
        return age_list
    if len(age_list) == 1:
        if 'child' in age_list[0] or 'adult' in age_list[0]:
            #print("Entered Child/Adult")
            age_list = age_list[0].split(',')
            age_list = [x.strip() for x in age_list]
            age_list = str(age_list).replace(']', '').replace('[', '').replace("'", "").replace(",", " |")
            return ['N/A', age_list]
    if len(age_list) == 2:
        ages = age_list[0].strip()
        categories = age_list[1].replace("(", "").replace(")", "").replace(",", " |")
        
        return [ages, categories]
        
    else:
        print("Invalid format", age_list)
        return ['Invalid', 'Invaid']



