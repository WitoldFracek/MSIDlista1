# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 1: Regresja liniowa
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
from typing import List, Any, Union

import numpy as np
from numpy import ndarray, number

from utils import polynomial


def mean_squared_error(x, y, w):
    """
    :param x: ciąg wejściowy Nx1
    :param y: ciąg wyjsciowy Nx1
    :param w: parametry modelu (M+1)x1
    :return: błąd średniokwadratowy pomiędzy wyjściami y oraz wyjściami
     uzyskanymi z wielowamiu o parametrach w dla wejść x
    """
    polinom = np.array(polynomial(x, w))
    es = ((y - polinom) ** 2)/np.shape(x)[0]
    return np.sum(es)


def design_matrix(x_train, M):
    """
    :param x_train: ciąg treningowy Nx1
    :param M: stopień wielomianu 0,1,2,...
    :return: funkcja wylicza Design Matrix Nx(M+1) dla wielomianu rzędu M
    """
    fi = np.hstack([x_train ** i for i in range(M+1)])
    return fi


def least_squares(x_train, y_train, M):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego 
    wielomianu, a err to błąd średniokwadratowy dopasowania
    """
    fi = design_matrix(x_train, M)
    fiT = np.copy(fi)
    fiT = np.transpose(fiT)
    fiTfi = fiT @ fi
    fiTfi = np.linalg.inv(fiTfi)
    first_part = fiTfi @ fiT
    w = first_part @ y_train
    err = mean_squared_error(x_train, y_train, w)
    ret = (w, err)
    return ret


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param M: rzad wielomianu
    :param regularization_lambda: parametr regularyzacji
    :return: funkcja zwraca krotkę (w,err), gdzie w są parametrami dopasowanego
    wielomianu zgodnie z kryterium z regularyzacją l2, a err to błąd 
    średniokwadratowy dopasowania
    """
    fi = design_matrix(x_train, M)
    fiT = np.copy(fi)
    fiT = np.transpose(fiT)
    fiTfi = fiT @ fi
    inversed = fiTfi + (np.identity(np.shape(fiTfi)[0]) * regularization_lambda)
    inversed = np.linalg.inv(inversed)
    w = (inversed @ fiT) @ y_train
    err = mean_squared_error(x_train, y_train, w)
    return w, np.sum(err)


def model_selection(x_train, y_train, x_val, y_val, M_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M_values: tablica stopni wielomianu, które mają byc sprawdzone
    :return: funkcja zwraca krotkę (w,train_err,val_err), gdzie w są parametrami
    modelu, ktory najlepiej generalizuje dane, tj. daje najmniejszy błąd na 
    ciągu walidacyjnym, train_err i val_err to błędy na sredniokwadratowe na 
    ciągach treningowym i walidacyjnym
    """
    w = None
    train_err = None
    val_err = None
    for M in M_values:
        w_temp, err_temp = least_squares(x_train, y_train, M)
        val_err_temp = mean_squared_error(x_val, y_val, w_temp)
        if val_err is None:
            val_err = val_err_temp
            train_err = err_temp
            w = w_temp
        elif val_err_temp < val_err:
            val_err = val_err_temp
            train_err = err_temp
            w = w_temp
    return w, train_err, val_err


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    """
    :param x_train: ciąg treningowy wejśćia Nx1
    :param y_train: ciąg treningowy wyjscia Nx1
    :param x_val: ciąg walidacyjny wejśćia Nx1
    :param y_val: ciąg walidacyjny wyjscia Nx1
    :param M: stopień wielomianu
    :param lambda_values: lista z wartościami różnych parametrów regularyzacji
    :return: funkcja zwraca krotkę (w,train_err,val_err,regularization_lambda),
    gdzie w są parametrami modelu, ktory najlepiej generalizuje dane, tj. daje
    najmniejszy błąd na ciągu walidacyjnym. Wielomian dopasowany jest wg
    kryterium z regularyzacją. train_err i val_err to błędy średniokwadratowe
    na ciągach treningowym i walidacyjnym. regularization_lambda to najlepsza
    wartość parametru regularyzacji
    """
    w = None
    train_err = None
    val_err = None
    regularization_lambda = None

    for la in lambda_values:
        w_temp, train_err_temp = regularized_least_squares(x_train, y_train, M, la)
        val_err_temp = mean_squared_error(x_val, y_val, w_temp)
        if val_err is None:
            val_err = val_err_temp
            train_err = train_err_temp
            w = w_temp
            regularization_lambda = la
        elif val_err_temp < val_err:
            val_err = val_err_temp
            train_err = train_err_temp
            w = w_temp
            regularization_lambda = la
    return w, train_err, val_err, regularization_lambda
