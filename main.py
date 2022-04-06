from flask import Flask, request, render_template
from keras.models import load_model
import pandas as pd
import joblib

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def compute():
    result = ''
    error = dict()
    if request.method == 'POST':
        data = request.form
        try:
            data1 = {k:[float(v)] for k,v in data.items()}
        except ValueError:
            for k,v in data.items():
                try:
                    float(v)
                except ValueError:
                    error['name'] = k
                    error['value'] = v
                    break
        if not error:
            input = pd.DataFrame(data=data1)
            res = scaler.transform(input)
            norm_input = pd.DataFrame(res, columns=input.columns)
            result = model.predict(norm_input)
            text_result = ', '.join(['{}={}'.format(y, x) for x, y in zip(result[0], result_cols)])
    return render_template('index.html', result=text_result, error=error)


if __name__ == '__main__':
    model = load_model('./new_model/')
    scaler = joblib.load('scaler.pkl')
    result_cols = ['Соотношение матрица-наполнитель', 'Модуль упругости при растяжении, ГПа',
              'Прочность при растяжении, МПа']
    app.run(host='localhost', port=8081, debug=True)
