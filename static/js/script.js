$(document).ready(function () {
    $('#upload-form').on('submit', function (event) {
        event.preventDefault();
        $('#results').html(''); // Clear previous results
        $('#progress-bars').html(''); // Clear existing progress bars

        var formData = new FormData(this);

        // Create a progress bar for each file
        for (var i = 0; i < this['images'].files.length; i++) {
            var file = this['images'].files[i];
            $('#progress-bars').append(`<div class="progress mb-2">
                                            <div class="progress-bar" role="progressbar" id="progress-bar-${i}" 
                                                 aria-valuemin="0" aria-valuemax="100" style="width: 0%"> 
                                                ${file.name}
                                            </div>
                                        </div>`);
        }

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
                        // Update progress bars
                        for (var i = 0; i < $('#upload-form')[0]['images'].files.length; i++) {
                            $(`#progress-bar-${i}`).css('width', percentComplete + '%').attr('aria-valuenow', percentComplete);
                        }
                        // Clear progress bars and show waiting message when upload is complete
                        if (percentComplete === 100) {
                            $('#progress-bars').html('');
                            $('#results').html(`
                                <div class="waiting-message">
                                    <div class="spinner-border text-primary" role="status"></div>
                                    <p class="message-text">Initiating Smart Merge...</p>
                                </div>
                            `);
                        }
                    }
                }, false);
                return xhr;
            },
            success: function (data) {
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
                                                  </div>`); // Removed the download link
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
