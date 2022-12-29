import numpy as np
from pandas import DataFrame, merge
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import contextlib
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import seaborn as sns
from tensorflow import convert_to_tensor

def split_df(a: DataFrame, test_prop: float):
    '''
    Divide un dataframe en tres (train, validation, test) segun la proporcion de test

    Parametros
    ----------
    a : un Dataframe
        Dataframe al que se dividira.
        
    test_prop : flotante
        Proporcion de la particion de test.

    Retorna
    -------
    splitting : una lista de dataframes de tamano 3 (train, validation, test)

    '''
    time = a.index.to_numpy()
    n = len(time)
    test_time = time[(int)(- n * (test_prop))]
    return a.loc[ : test_time].copy(), a.loc[test_time : ].copy()


def shift_data(df_x: DataFrame, df_y: DataFrame, n_steps_in: int, n_steps_out: int):
    '''
    Genera Caracteristicas pasadas y futuras desplazando por pasos de tiempos

    Parametros
    ----------
    df_x : DataFrame
        Dataframe con datos de entrada

    df_y : DataFrame
        Dataframe con datos de salida

    n_steps_in : entero
        Numero de pasos de tiempo pasados

    n_steps_out : entero
        Numero de pasos de tiempo futuros

    Retorna
    -------
    x, y : arrays
        Devuelve arrays de datos de entrada X, y datos de salida Y

    '''
    target_col = df_y.columns.to_numpy()
    features = df_x.columns.to_numpy()
    data_shifted = merge(df_x, df_y, left_index=True, right_index=True)
    x_cols, y_cols = list(), list()
    # Lag features
    for t in range(1, n_steps_in+1):
        data_shifted[features + '_t-' +
                     str(n_steps_in-t)] = data_shifted[features].shift(n_steps_in-t)
        x_cols = [*x_cols, *((features + '_t-'+str(n_steps_in-t)).tolist())]
    # Future feature
    data_shifted[target_col+'_t+' +
                    str(n_steps_out)] = data_shifted[target_col].shift(-n_steps_out)
    y_cols = [*y_cols, *((target_col + '_t+'+str(n_steps_out)).tolist())]

    data_shifted = data_shifted.dropna(how='any')
    x = data_shifted[x_cols].values
    x = convert_to_tensor(x.reshape(len(x), n_steps_in, len(features)))  # 3D
    y = convert_to_tensor(data_shifted[y_cols].values)  # 1D
    return x, y


def mae(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.mean(np.abs(orig - pred))


def rmse(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.sqrt(np.mean((orig - pred)**2))


def mape(orig, pred):
    # Verificar si tienen mismo shape
    if(orig.shape != pred.shape):
        raise Exception('Deben tener mismo shape')
    return np.mean(np.abs((orig - pred)/orig))*100

def plot_history(history,metric):
    fig, ax = plt.subplots(figsize = (8,5))
    ax.plot(history.history[metric],label='train '+metric)
    ax.plot(history.history['val_'+metric],label='validation '+metric)
    ax.legend()

def plot_pred(orig,pred,last_year,test_year=None):
    n = len(orig)
    y_o, y_p = orig.T, pred.T
    df = DataFrame()
    df['year'] = [last_year-(n-i+2) for i in range(n)]
    df['orig_1'] = y_o[0]
    df['orig_2'] = y_o[1]
    df['orig_3'] = y_o[2]
    df['pred_1'] = y_p[0]
    df['pred_2'] = y_p[1]
    df['pred_3'] = y_p[2]
    df = df.set_index('year')
    
    n = len(y_o)
    for i in range(n):
        fig, ax = plt.subplots(figsize=(8,4))
        line_o, = ax.plot(df['orig_'+str(i+1)],label='orig')
        line_p, = ax.plot(df['pred_'+str(i+1)],label='pred')
        ax.set_title('Predicción (t + {})'.format(i+1))
        ax.set_xlabel('Año (t)')
        ax.set_ylabel('Crecimiento PBI (%)')
        if test_year != None:
            ymin=np.minimum(df['orig_'+str(i+1)].min(),df['pred_'+str(i+1)].min())
            ymax=np.maximum(df['orig_'+str(i+1)].max(), df['pred_'+str(i+1)].max())
            plt.vlines(x = test_year, ymin = ymin, ymax = ymax ,colors = 'gray', ls = ':')
            plt.axvspan(test_year, last_year-3, facecolor="#ffcc66", alpha=0.5)
            test_r = mpatches.Patch(color='#ffcc66', label='Prueba')
            ax.legend(handles=[line_o,line_p,test_r])
        else:    
            ax.legend(handles=[line_o,line_p])
        plt.show()

def print_hp(path,tuner):
    with open(path,'a') as o:
        with contextlib.redirect_stdout(o):
            tuner.results_summary(num_trials=1)

def flatten(A : np.ndarray):
    r = list()
    for a in A:
        r.append(a.flatten())
    return np.array(r)

def print_line(line,path):
    f = open(path,'a')
    f.write(str(line))
    f.close()

def graficarTodo(df: DataFrame, t, linea_cero=False):
    fig, ax = plt.subplots(figsize=[12,8])
    df.drop(labels=['Class'], axis=1).plot(ax=ax)
    ax.set_title(t)
    y1, y2 = ax.get_ylim()
    resc1 = df['Class'] == 1
    ax.fill_between(resc1.index, y1=y1, y2=y2, where=resc1, facecolor='grey', alpha=0.4)
    if linea_cero:
        ax.axhline(y=0,color='grey',linestyle='--')
    plt.show()

def graficarClases(clase):
    neg, pos = np.bincount(clase)
    fig, ax = plt.subplots(figsize=[8,4])
    ax.set_title("Clases")
    ax.bar(['Neg','Pos'],[neg,pos])
    plt.show()

def plot_metrics(history):
    metrics = ['loss', 'auc', 'precision', 'recall']
    plt.subplots(figsize=[12,10])
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color='blue', label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color='blue', linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend();

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Expansiones Detectadas (Verdadero Negativos): ', cm[0][0])
    print('Expansiones Incorrecas Detectadas (Falso Positivos): ', cm[0][1])
    print('Recesiones no Detectadas (False Negativos): ', cm[1][0])
    print('Recesiones Detectadas (Verdadero Positivos): ', cm[1][1])
    print('Total de Recesiones: ', np.sum(cm[1]))

def plot_roc(train_labels, train_predictions, test_labels, test_predictions):
    train_fp, train_tp, _ = roc_curve(train_labels, train_predictions)
    test_fp, test_tp, _ = roc_curve(test_labels, test_predictions)
    _, ax = plt.subplots(figsize=[5,5])
    ax.plot(train_fp, train_tp, label="Train", color='blue')
    ax.plot(test_fp, test_tp, label="Test", color='blue', linestyle='--')
    ax.set_xlabel('False positives rate')
    ax.set_ylabel('True positives rate')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.legend(loc='lower right')
    ax.grid(True)

def plot_prc(train_labels, train_predictions, test_labels, test_predictions):
    train_precision, train_recall, _ = precision_recall_curve(train_labels, train_predictions)
    test_precision, test_recall, _ = precision_recall_curve(test_labels, test_predictions)

    _, ax = plt.subplots(figsize=[5,5])
    ax.plot(train_precision, train_recall, label="Train", color='blue')
    ax.plot(test_precision, test_recall, label="Test", color='blue', linestyle='--')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    ax.legend(loc='lower right')
    ax.grid(True)

def plot_probs(test_labels, test_predictions, tipo):
    _, ax = plt.subplots(figsize=[8,4])
    ax.plot(test_labels, label=f"{tipo} original", color='orange')
    ax.plot(test_predictions, label=f"{tipo} prediction", color='green')
    ax.set_xlabel('time steps')
    ax.set_ylabel('Probabilidad de Recesión')
    ax.legend()
    ax.grid(True)
    ax.legend(loc='lower right')