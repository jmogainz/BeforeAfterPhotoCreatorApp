$(document).ready(function() {
    $('#upload-form').on('submit', function(event) {
        event.preventDefault();
        var formData = new FormData(this);
        $.ajax({
            url: '/upload',  // Python Flask route
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            success: function(data) {
                $('#results').html('');
                if (data.links) {
                    data.links.forEach(link => {
                        $('#results').append(`<a href="${link}" download>Download Processed Image</a><br>`);
                    });
                } else {
                    $('#results').html('<p>No images processed.</p>');
                }
            },
            error: function() {
                $('#results').html('<p>An error occurred.</p>');
            }
        });
    });
});