from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Fieldset, Div, Row, Column, Submit
from crispy_forms.bootstrap import InlineField, PrependedText, InlineRadios


class TargetsFilterForm(forms.Form):
    form_type = forms.CharField(initial='filter', widget=forms.HiddenInput())
    query = forms.CharField(max_length=100, required=False, label="Filter Targets")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_action = 'targets'
        self.helper.form_show_labels = False
        self.helper.layout = Layout(
            'form_type',
            Row(
                InlineField(PrependedText('query', 'Filter:', placeholder='Search targets by names, titles or usernames')),
            )
        )


class TargetNewForm(forms.Form):
    form_type = forms.CharField(initial='new_target', widget=forms.HiddenInput())
    name = forms.CharField(max_length=150, required=True, label="Target name or coordinates")
    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_action = 'targets'
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('name', css_class="col-md-4"),
                Column('title', css_class="col-md-8"),
                css_class='align-items-end'
            ),
            Submit('submit', 'New target', css_class='btn-primary')
        )


class TargetInfoForm(forms.Form):
    form_type = forms.CharField(initial='target_info', widget=forms.HiddenInput())
    name = forms.CharField(max_length=150, required=False, label="Target name or coordinates")
    title = forms.CharField(max_length=150, required=False, label="Optional title or comment")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.field_template = 'crispy_field.html'
        self.helper.layout = Layout(
            'form_type',
            Row(
                Column('name', css_class="col-md-4"),
                Column('title', css_class="col-md-8"),
                css_class='align-items-end'
            ),
        )
