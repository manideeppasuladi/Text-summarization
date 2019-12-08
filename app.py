from flask import Flask,render_template,request
from TextSummary import TextSummarizer
app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/", methods = ['POST', 'GET'])
def mytext():
    if request.method == 'POST':
        text = request.form['textview']
        if request.form['num'] != "" :
            number = request.form['num']
        else:
            number = 5
        
        #app.logger.info(number)
        obj = TextSummarizer()
        mytext=obj.summarizeText(text,number)
        sent = ""
        for i in mytext:
            sent += i + " "
        return render_template('index.html',summary=sent)

if __name__ == '__main__': 
   app.run(debug = True) 
