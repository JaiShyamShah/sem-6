# webapp/views.py
from django.shortcuts import render
from django.http import HttpResponse
from datetime import date
import calendar
from calendar import HTMLCalendar

def index(request, year, month):
    year = int(year)  # Convert the year parameter to an integer
    month = int(month)  # Convert the month parameter to an integer
    
    # Check if the year is valid (between 1900 and 2099)
    if year < 1900 or year > 2099:
        year = date.today().year  # If invalid, use the current year

    # Get the name of the month from the calendar module
    month_name = calendar.month_name[month]
    
    # Create the title for the calendar page
    title = "MyClub Event Calendar - %s %s" % (month_name, year)
    
    # Generate the HTML calendar for the given year and month
    cal = HTMLCalendar().formatmonth(year, month)
    
    # Return the HTTP response with the calendar HTML
    return HttpResponse("<h1>%s</h1><p>%s</p>" % (title, cal))
