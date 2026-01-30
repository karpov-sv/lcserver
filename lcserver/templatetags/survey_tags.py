"""Template tags and filters for survey data sources."""

from django import template

register = template.Library()


@register.filter
def survey_condition_met(source_config, target):
    """Check if survey requirements are met for a target.
    
    Checks:
    - Coordinate requirements (target_ra/target_dec)
    - Name requirements (target_name)
    - Declination constraints (min/max)
    
    Usage in template:
        {% if source_config|survey_condition_met:target %}
    """
    config = target.config
    
    # Check coordinate requirement
    if source_config.get('requires_coordinates', True):
        if 'target_ra' not in config or 'target_dec' not in config:
            return False
    
    # Check name requirement (for sources like KWS that use names instead of coordinates)
    if not source_config.get('requires_coordinates', True):
        if 'target_name' not in config:
            return False
    
    # Check declination limits
    dec_min = source_config.get('declination_min')
    dec_max = source_config.get('declination_max')
    
    if dec_min is not None:
        target_dec = config.get('target_dec')
        if target_dec is None or target_dec < dec_min:
            return False
    
    if dec_max is not None:
        target_dec = config.get('target_dec')
        if target_dec is None or target_dec > dec_max:
            return False
    
    return True


@register.filter
def get_form(forms_dict, source_id):
    """Get form for a survey source from forms dictionary.

    The survey_forms dict is keyed by source_id (e.g., 'ztf'),
    so we just need to look up the source_id directly.

    Usage in template:
        {% with form=survey_forms|get_form:source_id %}
    """
    return forms_dict.get(source_id)
