import os
from flask import Flask, request, jsonify, render_template,redirect,make_response,url_for
import json
import re
#fimport request

app = Flask(_name_)

@app.route('/Crop')
#def running():
    #return 'Flask is running!'

def running():
    #message = request.get_json(force=True)
    crop_name = "Wheat"
    yield_value = "2.3"
    #response = {
      #'greeting': 'Hello,' + name + '!'  
    #}
    return crop_name + ' ' + yield_value
    #return "Flask is running!"
