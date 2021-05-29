import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import random
#from sklearn.linear_model import LinearRegression

app = Flask(__name__)
model = pickle.load(open('finalizedjscjcwjw_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
 
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    word_list1 = ['sea','sec','bee', 'see', 'tie', 'bag','leg','pig','log','zoo', 'sow', 'cow', 'six' , 'pay', 'key,buy']
    word_list2 = ['read','road','leaf','loan','soap','seat','boat','seal','feed','beef','feet','cool','pool','rain','coin','pain','gold','bold','horn','born','turn','worm','back','neck','kick','lock','bank','mask','desk','disk','pant','sent','wind','found','song','sing','fast','lost','dust','nest','list','cash','mesh','fish','push','ship','shop','pass','mess','losh']
    word_list3 = ['sam run','red cab','cow run','tea cup','big log','red gem','boy leg','red tie','big map']
    word_list4 = ['pink bag','good boy','sam sing']
    


    
    Form_age=int_features[0]
    output = round(prediction[0], 2)
    if output==1:
        abc=random.choice(word_list1)
    elif output==2:
        abc=random.choice(word_list2)
    elif output==3 and Form_age==7:
        abc=random.choice(word_list3)
    elif output==3 and Form_age==8:
        abc=random.choice(word_list4)

    return render_template('test.html', prediction_text='disable level {}'.format(output),abc='he will create --> {}'.format(abc),aaaa='test{}'.format(Form_age))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)