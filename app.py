from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import tensorflow as tf

app = Flask(__name__)

class Model():
    def __init__(self):
        # получаем массив уникальных скилов
        df = pd.read_csv('Data_uniq.csv', index_col=0)
        df = df.astype(int)
        self.uniq_skills = df.columns.values

        # получаем предсказывающую модель
        self.model = tf.keras.models.load_model('Salaries_predictor')

    # метод получения примера для предсказания
    def get_sample(self, custom_skills):
        vector = np.zeros(len(self.uniq_skills))
        for skill in custom_skills:
            for i, u_skill in enumerate(self.uniq_skills):
                if skill == u_skill:
                    vector[i] = 1
        return vector.reshape(1, len(self.uniq_skills))

    # метод получения предсказания
    def get_predict(self, sample):
        predict = self.model.predict(sample)
        return predict


model = Model()


@app.route('/', methods=['POST', 'GET'])   # указываем какой URL отслеживаем
def index():
    if request.method == 'POST':
        skills = request.form.getlist('mycheck')
        print(skills)
        print(type(skills))
        sample = model.get_sample(skills)
        predict = model.get_predict(sample)
    else:
        predict = '________'
#    salary = np.random.choice([111, 222, 333])

    return render_template('boxes.html', salary=predict[0][0])


@app.route('/feedback', methods=['POST', 'GET'])  # указываем какой URL отслеживаем
def feedback():
    if request.method == 'POST':
        skills = request.form.getlist('skill')
        salary = request.form.getlist('salary')
        print(type(skills), skills)
        print(type(salary), salary)

    return render_template('feedback.html')


if __name__ == '__main__':
    app.run(host='10.100.100.200', port='5666')