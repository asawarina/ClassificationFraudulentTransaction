# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:59:21 2020

@author: Guri
"""

import requests
data = requests.get("https://www.google.com/")
print(data.text)