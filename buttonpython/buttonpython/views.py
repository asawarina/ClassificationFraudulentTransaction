from django.shortcuts import render
import requests
import sys
from subprocess import run, PIPE
from django.core.files.storage import FileSystemStorage
import base64


def button(request):
    
   return render(request, 'home.html')

def output(request):
    
    data = requests.get("https://www.google.com/")
    print(data.text)

    data = data.text
    return render(request,'home.html',{'data':data})

def external(request):
    #inp= request.POST.get('param')
    image=request.FILES['image']

    print("image is ",image)

    fs=FileSystemStorage()
    filename=fs.save(image.name,image)
    fileurl=fs.open(filename)
    templateurl=fs.url(filename)

    print("file raw url",filename)
    print("file full url", fileurl)
    print("template url",templateurl)
   
    image= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//traditional.py',str(fileurl),str(filename)],shell=False,stdout=PIPE)
    
    print("image is ",image.stdout)
    
    return render(request,'home.html',{'raw_url':templateurl,'edit_url':image.stdout.decode("utf-8")})


def random(request):
    #inp= request.POST.get('param')
    image_random=request.FILES['image_random']

    print("image is ",image_random)

    fs_random=FileSystemStorage()
    filename_random=fs_random.save(image_random.name,image_random)
    fileurl_random=fs_random.open(filename_random)
    templateurl_random=fs_random.url(filename_random)

    print("file raw url",filename_random)
    print("file full url", fileurl_random)
    print("template url",templateurl_random)

    image_random= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//randomover.py',str(fileurl_random),str(filename_random)],shell=False,stdout=PIPE)
    
    print("image is ",image_random.stdout)
   
    return render(request,'home.html',{'raw_url_random':templateurl_random,'edit_url_random':image_random.stdout.decode("utf-8")})


def smote(request):
    #inp= request.POST.get('param')
    image_smote=request.FILES['image_smote']

    print("image is ",image_smote)

    fs_smote=FileSystemStorage()
    filename_smote=fs_smote.save(image_smote.name,image_smote)
    fileurl_smote=fs_smote.open(filename_smote)
    templateurl_smote=fs_smote.url(filename_smote)

    print("file raw url",filename_smote)
    print("file full url", fileurl_smote)
    print("template url",templateurl_smote)

    image_smote= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//smote.py',str(fileurl_smote),str(filename_smote)],shell=False,stdout=PIPE)
    
    print("image is ",image_smote.stdout)
    
    return render(request,'home.html',{'raw_url_smote':templateurl_smote,'edit_url_smote':image_smote.stdout.decode("utf-8")})

def pipeline(request):
    #inp= request.POST.get('param')
    image_pipe=request.FILES['image_pipe']

    print("image is ",image_pipe)

    fs_pipe=FileSystemStorage()
    filename_pipe=fs_pipe.save(image_pipe.name,image_pipe)
    fileurl_pipe=fs_pipe.open(filename_pipe)
    templateurl_pipe=fs_pipe.url(filename_pipe)

    print("file raw url",filename_pipe)
    print("file full url", fileurl_pipe)
    print("template url",templateurl_pipe)

    image_pipe= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//pipeline.py',str(fileurl_pipe),str(filename_pipe)],shell=False,stdout=PIPE)
    
    print("image is ",image_pipe.stdout)
    
    return render(request,'home.html',{'raw_url_pipe':templateurl_pipe,'edit_url_pipe':image_pipe.stdout.decode("utf-8")})


def randf(request):
    #inp= request.POST.get('param')
    image_randf=request.FILES['image_randf']

    print("image is ",image_randf)

    fs_randf=FileSystemStorage()
    filename_randf=fs_randf.save(image_randf.name,image_randf)
    fileurl_randf=fs_randf.open(filename_randf)
    templateurl_randf=fs_randf.url(filename_randf)

    print("file raw url",filename_randf)
    print("file full url", fileurl_randf)
    print("template url",templateurl_randf)

    image_randf= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//randomforest.py',str(fileurl_randf),str(filename_randf)],shell=False,stdout=PIPE)
    
    print("image is ",image_randf.stdout)
    
    return render(request,'home.html',{'raw_url_randf':templateurl_randf,'edit_url_randf':image_randf.stdout.decode("utf-8")})

def prcurve(request):
    #inp= request.POST.get('param')
    image_prcurve=request.FILES['image_prcurve']

    print("image is ",image_prcurve)

    fs_prcurve=FileSystemStorage()
    filename_prcurve=fs_prcurve.save(image_prcurve.name,image_prcurve)
    fileurl_prcurve=fs_prcurve.open(filename_prcurve)
    templateurl_prcurve=fs_prcurve.url(filename_prcurve)

    print("file raw url",filename_prcurve)
    print("file full url", fileurl_prcurve)
    print("template url",templateurl_prcurve)

    image_prcurve= run([sys.executable,'C://Users//asawari//Desktop//BackUp1//prcurve.py',str(fileurl_prcurve),str(filename_prcurve)],shell=False,stdout=PIPE)
    
    print("image is ",image_prcurve.stdout)
    
    return render(request,'home.html',{'raw_url_prcurve':templateurl_prcurve,'edit_url_prcurve':image_prcurve.stdout.decode("utf-8")})


def ensemble(request):
    #inp= request.POST.get('param')
    image_ensemble=request.FILES['image_ensemble']

    print("image is ",image_ensemble)

    fs_ensemble=FileSystemStorage()
    filename_ensemble=fs_ensemble.save(image_ensemble.name,image_ensemble)
    fileurl_ensemble=fs_ensemble.open(filename_ensemble)
    templateurl_ensemble=fs_ensemble.url(filename_ensemble)

    print("file raw url",filename_ensemble)
    print("file full url", fileurl_ensemble)
    print("template url",templateurl_ensemble)

    image_ensemble = run([sys.executable,'C://Users//asawari//Desktop//BackUp1//ensemble.py',str(fileurl_ensemble),str(filename_ensemble)],shell=False,stdout=PIPE)
    
    print("image is ",image_ensemble.stdout)
    
    return render(request,'home.html',{'raw_url_ensemble':templateurl_ensemble,'edit_url_ensemble':image_ensemble.stdout.decode("utf-8")})
