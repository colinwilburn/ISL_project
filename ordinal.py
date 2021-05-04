import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
import plotly.graph_objects as go

def implement_ridge(model, acc_true, err_true):
    acc_list = []
    err_list = []
    alpha_list = list(np.logspace(-3, 4))
    alpha_max = 0
    acc_max = 0
    for alpha in alpha_list:
        model.alpha = alpha
        acc, acc_err = basic_analysis(model, make_plots=False)
        acc_list.append(acc)
        err_list.append(acc_err)
        if acc > acc_max and acc > acc_true: #acc_true - acc < err_true: 
            alpha_max = alpha
            acc_max = acc
    
    model.alpha=alpha_max
    title=f'Ordinal, α={round(alpha_max, 2)}'
    acc_alpha, err_alpha = basic_analysis(
        model, make_plots=True, title=title
    )
    print(f"Best ridge regression with alpha={alpha_max}, acc={acc_alpha}, sd={err_alpha}")


    fig_ridge = go.Figure()
    fig_ridge.add_trace(go.Scatter(
        x = alpha_list,
        y = acc_list,
        error_y = dict(type='data', array=err_list),
        mode='markers',
        showlegend=False
    ))
    fig_ridge.add_trace(go.Scatter(
        x = [alpha_max],
        y = [acc_alpha],
        error_y = dict(type='data', array=[err_alpha]),
        mode='markers',
        name='Highest CV Accuracy'
    ))
    fig_ridge.update_xaxes(type="log")
    fig_ridge.update_layout(
        xaxis_title='α',
        yaxis_title='CV Accuracy',
        title="Ordinal, ridge regression",
        title_x=0.5,
        title_y=0.97,
        font=dict(family="Courier New, monospace",size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    fig_ridge.show(renderer='firefox')
    return acc_alpha, err_alpha


def basic_analysis(model, make_plots=True, title=None):
    target, features = build_training()
    acc_score = make_scorer(acc_fun)
    cv = cross_validate(
        estimator = model,
        X = features,
        y = target,
        cv = 10,
        scoring = acc_score,
        return_estimator = True
    ) # 10-fold cross validation scoring by accuracy
    acc = np.mean(cv['test_score']) # the average accuracy over the 10 folds
    acc_err = 2 * np.std(cv['test_score']) # the std error of the accuracy over 10 folds
    model.fit(features, target)
    if make_plots:
        make_coefficients_plot(cv, model, features, title)
        make_ratings_plot(target, features, model, cv, title)
        #make_desc_hist(features)
    return acc, acc_err
    


def build_training():
    df = pd.read_csv("Video_games_esrb_rating.csv") # read in the cvs
    features = df.iloc[:, 1:-1] # create data matrix, exclude rating and title
    
    ratings_map = {
        'E': 0, 'ET': 1, 'T': 2, 'M': 3
    } ###CHANGED
    target = df['esrb_rating'] # the response variable: the rating
    num_list = []
    for rating in target: # replace ratings with numerical values from 0-5
        num = ratings_map[rating]
        num_list.append(num)
    target = pd.Series(num_list)

    return target, features

def build_test():
    df = pd.read_csv("test_esrb.csv") # read in the cvs
    features = df.iloc[:, 1:-1] # create data matrix, exclude rating and title
    
    ratings_map = {
        'E': 0, 'ET': 1, 'T': 2, 'M': 3
    }
    target = df['esrb_rating'] # the response variable: the rating
    num_list = []
    for rating in target: # replace ratings with numerical values from 0-5
        num = ratings_map[rating]
        num_list.append(num)
    target = pd.Series(num_list)

    return target, features


def acc_fun(target_true, target_fit):
    target_fit = np.round(target_fit)
    target_fit.astype('int')
    return accuracy_score(target_true, target_fit)

def make_coefficients_plot(cv, model, features, title):
    coeffs_arr = np.zeros((32, 10))
    for i, model_cv in enumerate(cv['estimator']):
        for j, val in enumerate(model_cv.coef_):
            coeffs_arr[j][i] = val
    coeffs_val = []
    coeffs_err = []
    for row in coeffs_arr:
        coeffs_val.append(np.mean(row))
        coeffs_err.append(2 * np.std(row))

    delete_list = []
    coeffs_abs = [abs(i) for i in coeffs_val]
    coeffs_sorted = sorted(range(len(coeffs_abs)), key=lambda x: coeffs_abs[x])
    for rank, index in enumerate(coeffs_sorted):
        if rank < (32-15):
            delete_list.append(index)

    vals = np.delete(coeffs_val, delete_list)
    errs = np.delete(coeffs_err, delete_list)
    desc = np.delete(np.array(features.columns), delete_list)
    vals_full = np.delete(np.array(model.coef_), delete_list)


    fig_coeffs = go.Figure()
    fig_coeffs.add_trace(go.Bar(
        x = desc,
        y = vals,
        error_y = dict(type='data', array=errs),
        name='CV estimates',
    ))
    fig_coeffs.add_trace(go.Scatter(
        x=desc,
        y=vals_full,
        mode='markers',
        name='Full train estimates'
    ))
    fig_coeffs.update_layout(
        yaxis_title='Coefficient Value',
        title=title,
        title_x=0.5,
        title_y=0.97,
        font=dict(family="Courier New, monospace",size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig_coeffs.show(renderer='firefox')

def make_ratings_plot(target, features, model, cv, title):
    rating_names = ['E', 'ET', 'T', 'M']
    true_arr = acc_summary(target) # the proportions assigned to each rating in true set
    pred_arr = acc_summary(model.predict(features)) # proportions for prediction

    cv_arr = np.zeros((6, 10)) # now we get the average proportions for cv predictions
    for i, model_cv in enumerate(cv['estimator']):
        cv_model = acc_summary(model_cv.predict(features))
        for j, val in enumerate(cv_model):
            cv_arr[j][i] += val
    cv_val = []
    cv_err = []
    for row in cv_arr:
        cv_val.append(np.mean(row))
        cv_err.append(2 * np.std(row))

    fig_pred = go.Figure()
    fig_pred.add_trace(go.Bar(
        x = rating_names,
        y= true_arr,
        name='True Train'
    ))
    fig_pred.add_trace(go.Bar(
        x=rating_names,
        y=pred_arr,
        name='Full Train'
    ))
    fig_pred.add_trace(go.Bar(
        x=rating_names,
        y=cv_val,
        name='CV Train',
        error_y = dict(type='data', array=cv_err)
    ))
    fig_pred.update_layout(
        yaxis_title='Count',
        title=title,
        title_x=0.5,
        title_y=0.97,
        font=dict(family="Courier New, monospace",size=14),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    fig_pred.show(renderer='firefox')


def acc_summary(targets_list):
    pred_list = [0, 0, 0, 0]
    for pred in targets_list:
        pred_list[pred] += 1
    return pred_list


def make_desc_hist(features):
    counts = []
    for col in features.columns:
        counts.append(0)
    for index, row in features.iterrows():
        for i, column in enumerate(list(features.columns)):
            counts[i] += row[column]

    fig_desc = go.Figure()
    fig_desc.add_trace(go.Bar(
        x=features.columns,
        y=counts,
    ))
    fig_desc.update_layout(
        yaxis_title='Count',
        title='Distribution of Predictors in Train Set',
        title_x=0.5,
        title_y=0.97,
        font=dict(family="Courier New, monospace",size=14),
    )
    fig_desc.show(renderer='firefox')


    
