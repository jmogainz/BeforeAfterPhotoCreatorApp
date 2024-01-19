$(document).ready(function () {
    $('#upload-form').on('submit', function (event) {
        event.preventDefault();
        $('#results').html(''); // Clear previous results
        $('#progress-bars').html(''); // Clear existing progress bars

        var formData = new FormData(this);
        var fileCount = this['images'].files.length; // Get the count of files

        // Create a single progress bar
        $('#progress-bars').append(`<div class="progress mb-2">
                                        <div class="progress-bar" role="progressbar" id="total-progress-bar" 
                                             aria-valuemin="0" aria-valuemax="100" style="width: 0%"> 
                                            Uploading Photos...
                                        </div>
                                    </div>`);

        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            xhr: function () {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener("progress", function (evt) {
                    if (evt.lengthComputable) {
                        var percentComplete = Math.round((evt.loaded / evt.total) * 100);
                        // Update the single progress bar
                        $('#total-progress-bar').css('width', percentComplete + '%').attr('aria-valuenow', percentComplete);

                        // Change message based on file count when upload is complete
                        if (percentComplete === 100) {
                            $('#progress-bars').html('');
                            var message = fileCount > 15 ? 
                                "Attempting Smart Merge... <br>Lots of files were uploaded... <br>May be a dumb merge..." :
                                "Performing Smart Merge...";
                            $('#results').html(`
                                <div class="waiting-message">
                                    <div class="spinner-border text-primary" role="status"></div>
                                    <p class="message-text">${message}</p>
                                </div>
                            `);
                        }
                    }
                }, false);
                return xhr;
            },
            success: function (data) {
                $('#results').html(''); // Clear previous messages

                if (data.error && data.error === 'timestamp_missing') {
                    $('#results').append(`<p>Error: ${data.message}</p>`);
                    return;
                }
                // Display the "Smart Merge Complete!" message
                $('#results').html(`
                    <div class="waiting-message">
                        <p class="message-text">Smart Merge Complete!</p>
                    </div>
                `);

                // Set a timeout to wait before displaying the images
                setTimeout(function () {
                    $('#results').html(''); // Clear the "Smart Merge Complete!" message
                    // Process results
                    if (data.links) {
                        data.links.forEach(link => {
                            $('#results').append(`<div class="image-container">
                                                    <img src="${link}" alt="Processed Image" style="width:100%;max-width:300px;">
                                                  </div>`); 
                        });
                    } else {
                        $('#results').html('<p>No images processed.</p>');
                    }
                }, 2000); // 2000 milliseconds delay
            },
            error: function () {
                $('#results').html('<p>An error occurred.</p>');
            }
        });
    });
});
