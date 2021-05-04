import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression

from ordinal import build_training, build_test, acc_fun

def implement_lasso(acc_true, err_true):
    acc_list = []
    err_list = []
    c_list = list(np.logspace(4, -4, 50)) # change to (4, -4, 50) for final run
    alpha_list = []
    acc_max = 0
    c_max = c_list[0]
    for c in c_list:
        model = LogisticRegression(
            penalty='l1', solver='saga', multi_class='multinomial', max_iter=1e5,
            random_state=1, C=c
        )
        print(1 / c)
        acc, acc_err = basic_analysis(model)
        acc_list.append(acc)
        err_list.append(acc_err)
        alpha_list.append(1 / c)
        if acc > acc_true and acc > acc_max: #acc_true - acc < err_true:
            c_max = c
            acc_max = acc

    model = LogisticRegression(
        penalty='l1', solver='saga', multi_class='multinomial', max_iter=1e5,
        random_state=1, C=c_max
    )
    title = f"Multinomial, α={round(1/c_max, 2)}"
    acc_alpha, err_alpha = basic_analysis(model, make_plots=False, title=title) # for now
    print(f"Best lasso regression with alpha={1 / c_max}, acc={acc_alpha}, sd={err_alpha}")
    print(f"Test accuracy w/ alpha={1 / c_max}:    {get_test_acc(model)}")

    fig_lasso = go.Figure()
    fig_lasso.add_trace(go.Scatter(
        x = alpha_list,
        y = acc_list,
        error_y = dict(type='data', array=err_list),
        mode='markers'
    ))
    fig_lasso.add_trace(go.Scatter(
        x = [1 / c_max],
        y = [acc_alpha],
        error_y = dict(type='data', array=[err_alpha]),
        mode='markers',
        name='Highest CV Accuracy'
    ))
    fig_lasso.update_layout(
        xaxis_title='α',
        yaxis_title='CV Accuracy',
        title="Multinomial, lasso regression",
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
    fig_lasso.update_xaxes(type="log")
    fig_lasso.show(renderer='firefox')


def basic_analysis(model, make_plots=False):
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
        title = f"Multinomial, α=0"
        make_coefficients_plot(cv, model, features, title)
        make_ratings_plot(target, features, model, cv, title)
    return acc, acc_err

def make_coefficients_plot(cv, model, features, title):
    coeffs_arr = np.zeros((4, 32, 10))
    for i, model_cv in enumerate(cv['estimator']): # each of 10 cv models
        for j, coef_list in enumerate(model_cv.coef_): # the coefficients for each rating
            for k, coef in enumerate(coef_list):
                coeffs_arr[j][k][i] = coef
    
    coeffs_val = np.zeros((4, 32))
    coeffs_err = np.zeros((4, 32))
    for rating, coeffs in enumerate(coeffs_arr):
        for i, coeff_list in enumerate(coeffs):
            print(coeff_list)
            coeffs_val[rating][i] = np.mean(coeff_list)
            coeffs_err[rating][i] = 2 * np.std(coeff_list)

    coeffs_val_new = []
    coeffs_err_new = []
    delete_list = [[], [], [], []]
    desc = []
    vals_full = []
    print(model.C)
    for rating in range(4):
        coeffs_abs = [abs(i) for i in coeffs_val[rating]]
        coeffs_sorted = sorted(range(len(coeffs_abs)), key=lambda x: coeffs_abs[x])
        for rank, index in enumerate(coeffs_sorted):
            if rank < (32 - 5):
                delete_list[rating].append(index)
        coeffs_val_new.append(np.delete(coeffs_val[rating], delete_list[rating]))
        coeffs_err_new.append(np.delete(coeffs_err[rating], delete_list[rating]))
        desc.append(np.delete(np.array(features.columns), delete_list[rating]))
        vals_full.append(np.delete(np.array(model.coef_[rating]), delete_list[rating]))
    coeffs_val = coeffs_val_new
    coeffs_err = coeffs_err_new

    rows = [1, 1, 2, 2]
    cols = [1, 2, 1, 2]
    rating_names = ['E', 'ET', 'T', 'M']
    fig_lasso = make_subplots(rows=2, cols=2)
    for rating, vals in enumerate(coeffs_val):
        errs = coeffs_err[rating]
        var_names = desc[rating]
        fig_lasso.add_trace(
            go.Bar(
                x=var_names,
                y=vals,
                error_y = dict(type='data', array=errs),
                name=f'{rating_names[rating]} CV estimates'
            ), row=rows[rating], col=cols[rating]
        )
        fig_lasso.add_trace(
            go.Scatter(
                x=var_names,
                y=vals_full[rating],
                name=f'{rating_names[rating]} Full train estimates',
                mode='markers'
            ), row=rows[rating], col=cols[rating]
        )
        fig_lasso.update_yaxes(title_text='Coefficient Value', row=rows[rating], col=cols[rating])
        fig_lasso.update_xaxes(tickangle=-20, row=rows[rating], col=cols[rating])

    fig_lasso.update_layout(
        title=title,
        title_x=0.5,
        title_y=0.97,
        font=dict(family="Courier New, monospace",size=14),
        xaxis_tickangle=-20
    )
    fig_lasso.show(renderer='firefox')

def make_ratings_plot(target, features, model, cv, title):
    rating_names = ['E', 'ET', 'T', 'M']
    true_arr = acc_summary(target)
    pred_arr = acc_summary(model.predict(features))
    cv_arr = np.zeros((4, 10))
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

def get_test_acc(model):
    target, features = build_test()
    pred = model.predict(features)
    num_correct = 0
    for i, rating in enumerate(pred):
        if rating == target[i]:
            num_correct += 1
    return num_correct / len(target)

