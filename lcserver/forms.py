from django import forms

from crispy_forms.helper import FormHelper
from crispy_forms.layout import Layout, Field, Row, Column, Submit
from crispy_forms.bootstrap import InlineField, PrependedText, InlineRadios

from . import surveys
from . import processing  # Import to trigger decorator registration


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


def create_survey_form(source_id, survey_config):
    """Factory function to create a Django form for a survey source."""

    # Special case for 'info' form which has extra fields
    if source_id == 'info':
        class TargetInfoForm(forms.Form):
            form_type = forms.CharField(initial='target_info', widget=forms.HiddenInput())
            name = forms.CharField(max_length=150, required=False, label="Target name or coordinates")
            title = forms.CharField(max_length=150, required=False, label="Optional title or comment")
            g_minus_r = forms.FloatField(required=False, label="(g - r) color")
            B_minus_V = forms.FloatField(required=False, label="(B - V) color")

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
                    Row(
                        Column('g_minus_r', css_class="col-md"),
                        Column('B_minus_V', css_class="col-md"),
                        css_class='align-items-end'
                    ),
                )
        return TargetInfoForm

    # Build field dict starting with form_type
    fields = {
        'form_type': forms.CharField(
            initial=f'target_{source_id}',
            widget=forms.HiddenInput()
        )
    }

    # Add custom fields from registry
    layout_fields = ['form_type']
    for field_name, field_config in survey_config.get('form_fields', {}).items():
        if field_config['type'] == 'choice':
            if source_id == 'ztf':
                # Special case: ZTF uses RadioSelect widget
                fields[field_name] = forms.ChoiceField(
                    label=field_config['label'],
                    choices=field_config['choices'],
                    initial=field_config['initial'],
                    required=field_config['required'],
                    widget=forms.RadioSelect
                )
            else:
                fields[field_name] = forms.ChoiceField(
                    label=field_config['label'],
                    choices=field_config['choices'],
                    initial=field_config['initial'],
                    required=field_config['required']
                )
        elif field_config['type'] == 'float':
            fields[field_name] = forms.FloatField(
                label=field_config['label'],
                initial=field_config['initial'],
                required=field_config['required']
            )
        layout_fields.append(field_name)

    # Create form class dynamically
    FormClass = type(
        f'Target{source_id.upper()}Form',
        (forms.Form,),
        fields
    )

    # Define __init__ to add crispy form helper
    def form_init(self, *args, **kwargs):
        super(FormClass, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_tag = False
        self.helper.field_template = 'crispy_field.html'

        # Special layout for ZTF with InlineRadios
        if source_id == 'ztf' and 'ztf_color_model' in layout_fields:
            self.helper.layout = Layout(
                'form_type',
                Row(
                    Column(InlineRadios('ztf_color_model', template='crispy_radioselect_inline.html'), css_class='form-group'),
                    css_class='align-items-end'
                ),
            )
        else:
            self.helper.layout = Layout(*layout_fields)

    FormClass.__init__ = form_init
    return FormClass


# Auto-generate forms for all survey sources
_survey_forms = {}
for source_id, config in surveys.SURVEY_SOURCES.items():
    # Skip sources without processing functions (lightcurve-only sources)
    if config.get('processing_function') is None:
        continue
    _survey_forms[source_id] = create_survey_form(source_id, config)

# Named references for backward compatibility
TargetInfoForm = _survey_forms['info']
TargetZTFForm = _survey_forms['ztf']
TargetASASForm = _survey_forms['asas']
TargetCSSForm = _survey_forms['css']
TargetTESSForm = _survey_forms['tess']
TargetDASCHForm = _survey_forms['dasch']
TargetAPPLAUSEForm = _survey_forms['applause']
TargetMMT9Form = _survey_forms['mmt9']
TargetCombinedForm = _survey_forms['combined']


def get_survey_form(source_id):
    """Get form class for a survey source."""
    return _survey_forms.get(source_id)
