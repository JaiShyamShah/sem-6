from django.shortcuts import render
from .forms import StudentForm

def student_form(request):
    if request.method == "POST":
        form = StudentForm(request.POST)
        if form.is_valid():
            # Calculate total marks and percentage
            marks_english = form.cleaned_data['marks_english']
            marks_physics = form.cleaned_data['marks_physics']
            marks_chemistry = form.cleaned_data['marks_chemistry']
            total_marks = marks_english + marks_physics + marks_chemistry
            total_percentage = (total_marks / 300) * 100

            # Prepare student details for textarea display
            student_details = {
                'name': form.cleaned_data['name'],
                'dob': form.cleaned_data['dob'],
                'address': form.cleaned_data['address'],
                'contact_number': form.cleaned_data['contact_number'],
                'email': form.cleaned_data['email'],
                'marks_english': marks_english,
                'marks_physics': marks_physics,
                'marks_chemistry': marks_chemistry,
                'total_percentage': total_percentage
            }
            
            return render(request, 'student_app/student_form.html', {'form': form, 'student_details': student_details})
    else:
        form = StudentForm()

    return render(request, 'student_app/student_form.html', {'form': form})
