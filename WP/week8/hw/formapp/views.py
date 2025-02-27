from django.shortcuts import render

# Grocery items with prices
GROCERY_ITEMS = {
    "Wheat": 40,
    "Jaggery": 60,
    "Dal": 80
}

def index(request):
    # Initialize session if not already set
    if "selected_items" not in request.session:
        request.session["selected_items"] = []

    selected_items = request.session["selected_items"]

    if request.method == "POST":
        selected_names = request.POST.getlist("items")  # Get checked items
        
        # Add new items only if they are not already in the session
        for item in selected_names:
            if item not in [i[0] for i in selected_items]:
                selected_items.append((item, GROCERY_ITEMS[item]))

        # Update session
        request.session["selected_items"] = selected_items
        request.session.modified = True  # Ensures session updates

    return render(request, "index.html", {
        "grocery_items": GROCERY_ITEMS,
        "selected_items": selected_items
    })
