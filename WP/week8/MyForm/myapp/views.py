from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.sessions.models import Session

def first_page(request):
    if request.method == "POST":
        # Capture the form data and store it in the session
        name = request.POST.get('name')
        roll = request.POST.get('roll')
        subject = request.POST.get('subject')
        
        # Store values in session
        request.session['name'] = name
        request.session['roll'] = roll
        request.session['subject'] = subject
        
        # Redirect to the second page
        return redirect('second_page')

    return render(request, 'firstPage.html')

def second_page(request):
    # Retrieve session data
    name = request.session.get('name')
    roll = request.session.get('roll')
    subject = request.session.get('subject')
    
    # If no session data exists, return to the first page
    if not name or not roll or not subject:
        return redirect('first_page')
    
    return render(request, 'secondPage.html', {'name': name, 'roll': roll, 'subject': subject})

# Create your views here.
