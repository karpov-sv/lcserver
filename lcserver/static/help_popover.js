$(document).ready(function(){
    // Initialize popovers from spans
    $('.span-help-popover').each(function(index) {
        var x=$(this);
        // In the colours of the page around it where it asks for that
        var secondary = x.hasClass('help-secondary');
        x.addClass("fa fa-question-circle small");
        x.addClass(secondary ? "text-secondary" : "text-info");
        x.attr('data-bs-trigger', 'hover focus');
        // x.attr('data-bs-placement', 'left');
        x.attr('data-bs-custom-class', secondary ? 'help-popover help-popover-secondary' : 'help-popover');
        x.attr('data-bs-content', x.html());
        x.html(''); // Remove the text from span itself, as it is in the popover now.

        const popover = new bootstrap.Popover(x, {html: true, container: 'body'});
    });
});
