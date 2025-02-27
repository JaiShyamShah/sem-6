from django import forms

class CalculatorForm(forms.Form):
    number1 = forms.IntegerField(label='First Number')
    number2 = forms.IntegerField(label='Second Number')
    operation = forms.ChoiceField(
        label='Operation',
        choices=[('+', 'Addition'), ('-', 'Subtraction'), ('*', 'Multiplication'), ('/', 'Division')]
    )
