from django.shortcuts import render

def magazine_cover(request):
    context = {}
    if request.method == 'POST':
        context['image'] = request.POST.get('image')
        context['bg_color'] = request.POST.get('bg_color')
        context['font_size'] = request.POST.get('font_size')
        context['font_color'] = request.POST.get('font_color')
        context['font_family'] = request.POST.get('font_family')
        context['message'] = request.POST.get('message')
    
    return render(request, 'index.html', context)

