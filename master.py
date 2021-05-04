from mord import LogisticAT
from sklearn.linear_model import LogisticRegression
import ordinal
import multi
import pandas as pd
'''
model_ordinal = LogisticAT(alpha=0)

ord_acc, ord_err = ordinal.basic_analysis(
    model_ordinal, make_plots=True, title=f'Ordinal, α={model_ordinal.alpha}'
)
print(f"Avg 10-fold CV accuracy for ordinal w/ α={model_ordinal.alpha}:    {ord_acc} +/- {ord_err}")
print(f"Test accuracy for ordinal w/ α={model_ordinal.alpha}:    {multi.get_test_acc(model_ordinal)}")


ridge_acc, ridge_err = ordinal.implement_ridge(model_ordinal, ord_acc, ord_err)
print(f"Avg 10-fold CV accuracy for ordinal w/ α={model_ordinal.alpha}:    {ord_acc} +/- {ord_err}")
print(f"Test accuracy for ordinal w/ α={model_ordinal.alpha}:    {multi.get_test_acc(model_ordinal)}")
'''

model_multi = LogisticRegression(
    penalty='none', solver='lbfgs', multi_class='multinomial', max_iter=1e5
) 
multi_acc, multi_err = multi.basic_analysis(model_multi, make_plots=True)
print(f"Avg 10-fold CV accuracy for multinomial:    {multi_acc} +/- {multi_err}")
print(f"Test accuracy for multinomial:    {multi.get_test_acc(model_multi)}")


multi.implement_lasso(multi_acc, multi_err)
print(f"Avg 10-fold CV accuracy for multinomial:    {multi_acc} +/- {multi_err}")







