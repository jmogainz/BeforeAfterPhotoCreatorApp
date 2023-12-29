$(document).ready(function() {
    $('#upload-form').on('submit', function(event) {
        event.preventDefault();
        $('#results').html(''); // Clear previous results
        $('#loader').show(); // Show the spinner

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
                        $('#results').append(`<div class="image-container">
                                                <img src="${link}" alt="Processed Image" style="width:100%;max-width:300px;">
                                                <a href="${link}" download>Download</a>
                                              </div>`);
                    });
                } else {
                    $('#results').html('<p>No images processed.</p>');
                }
            },
            error: function() {
                $('#results').html('<p>An error occurred.</p>');
            },
            complete: function() {
                $('#loader').hide(); // Hide the spinner
            }
        });
    });
});
