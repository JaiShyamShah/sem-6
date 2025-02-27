from django.shortcuts import render, redirect
from .forms import CarForm

def car_form_view(request):
    if request.method == "POST":
        form = CarForm(request.POST)
        if form.is_valid():
            manufacturer = form.cleaned_data['manufacturer']
            model = form.cleaned_data['model']
            # Redirect to the result page with parameters passed via query string.
            return redirect(f'/result/?manufacturer={manufacturer}&model={model}')
    else:
        form = CarForm()
    return render(request, 'car_form.html', {'form': form})

def result_view(request):
    # Retrieve parameters from the query string.
    manufacturer = request.GET.get('manufacturer', 'Not provided')
    model = request.GET.get('model', 'Not provided')
    context = {
        'manufacturer': manufacturer,
        'model': model
    }
    return render(request, 'result.html', context)