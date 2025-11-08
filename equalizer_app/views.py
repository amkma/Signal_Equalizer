from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def human(request):
    return render(request, 'human.html')

def music(request):
    return render(request, 'music.html')

def animal(request):
    return render(request, 'animal.html')
