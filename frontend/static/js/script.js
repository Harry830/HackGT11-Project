$(document).ready(function () {
    // Send message on button click or enter key
    $('#sendButton').click(sendMessage);
    $('#questionInput').keypress(function (e) {
        if (e.which === 13 && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    function sendMessage() {
        var question = $('#questionInput').val().trim();
        if (question === '') return;
        $('#questionInput').val('');
        $('#chatBox').append(
            '<div class="message user"><div class="text">' + question + '</div></div>'
        );
        $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);

        // Show typing indicator
        $('#chatBox').append(
            '<div class="message assistant" id="typingIndicator"><div class="text">Typing...</div></div>'
        );
        $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);

        $.ajax({
            url: '/ask',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question: question }),
            success: function (data) {
                $('#typingIndicator').remove();
                $('#chatBox').append(
                    '<div class="message assistant"><div class="text">' + data.response + '</div></div>'
                );
                $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);
            },
            error: function () {
                $('#typingIndicator').remove();
                $('#chatBox').append(
                    '<div class="message assistant"><div class="text">Error: Could not process the question.</div></div>'
                );
                $('#chatBox').scrollTop($('#chatBox')[0].scrollHeight);
            }
        });
    }

    // File Upload with Progress
    $('#uploadForm').on('submit', function (e) {
        e.preventDefault();
        var formData = new FormData(this);
        $('.progress').show();  // Show the progress bar
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            contentType: false,
            processData: false,
            xhr: function () {
                var xhr = new window.XMLHttpRequest();
                xhr.upload.addEventListener('progress', function (evt) {
                    if (evt.lengthComputable) {
                        var percentComplete = (evt.loaded / evt.total) * 100;
                        $('.progress-bar').width(percentComplete + '%');
                        $('.progress-bar').attr('aria-valuenow', percentComplete);
                    }
                }, false);
                return xhr;
            },
            success: function (data) {
                $('#uploadResponse').html('<p class="text-success">' + data.response + '</p>');
                updateFileList();  // Update file list after upload
                $('.progress-bar').width('0%');  // Reset progress bar
                $('.progress').hide();  // Hide the progress bar
                $('#uploadModal').modal('hide');  // Close modal
            },
            error: function (xhr) {
                $('#uploadResponse').html('<p class="text-danger">' + xhr.responseJSON.response + '</p>');
                $('.progress').hide();  // Hide the progress bar
            }
        });
    });

    // Add new scenario
    $('#addScenarioForm').on('submit', function (e) {
        e.preventDefault();
        var scenario = $('#scenarioInput').val();
        $.ajax({
            url: '/add_scenario',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ scenario: scenario }),
            success: function (data) {
                $('#scenarioResponse').html('<p class="text-success">' + data.response + '</p>');
                $('#scenarioInput').val('');
                $('#scenarioModal').modal('hide');  // Close modal
            },
            error: function () {
                $('#scenarioResponse').html('<p class="text-danger">Error adding new scenario</p>');
            }
        });
    });

    // Function to update file list
    function updateFileList() {
        $.ajax({
            url: '/files',
            type: 'GET',
            success: function (data) {
                var fileListHtml = '<ul class="list-group list-group-flush">';
                data.files.forEach(function (file) {
                    fileListHtml += '<li class="list-group-item bg-dark text-white">' + file + '</li>';
                });
                fileListHtml += '</ul>';
                $('#fileList').html(fileListHtml);
            }
        });
    }

    // Load the file list on page load
    updateFileList();
});
