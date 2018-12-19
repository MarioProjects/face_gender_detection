$(document).ready(function () {
    // Init
    $('.image-section-uploaded').hide(); // We hide this div since no image has been uploaded yet
    $('.loaderModel').hide(); // It is not predicting in the model so we hide the loader
    $('#resultModel').hide(); // We can not have the result of the model -> We have just loaded the page!

    // Upload Image Preview
    function readURLImg(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                var myCanvas = document.getElementById('canvasPreview');
                var ctx = myCanvas.getContext('2d');
                var img = new Image;
                img.src = e.target.result;
                img.onload = function() {
                    // access image size here
                    myCanvas.width = this.width;
                    myCanvas.height = this.height;
                    ctx.drawImage(img,0,0); // Or at whatever offset you like
                    callModel();
                };
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    // We register when the image has been uploaded
    $("#imageUpload").change(function () {
        $('.image-section-uploaded').show(); // We show the image in a div/canvas
        // We hide the possible previous result of another image
        $('#resultModel').text(''); // We delete possible previous prediction (text)
        $('#resultModel').hide(); // We hide the div with the text of the prediction (it has not yet been done)
        readURLImg(this);
    });

});

function callModel(){
    var form_data = new FormData($('#upload-file-model')[0]); // We take the data that has been uploaded

    // Show loading animation
    $(this).hide();
    $('.loaderModel').show();

    // Make prediction by calling api /predictModel
    $.ajax({
        type: 'POST',
        url: '/predictModel',
        data: form_data,
        contentType: false,
        cache: false,
        processData: false,
        async: true,
        success: function (data) {
            data = JSON.parse(data);
            // Get and display the result
            var myCanvas = document.getElementById('canvasPreview');
            var ctx = myCanvas.getContext('2d');

            if (data.faces!=undefined){
                for (var face in data.faces){
                    face = data.faces[face] // 'face' toma el valor de la clave para poder acceder al objeto
                    ctx.lineWidth = 2;
                    ctx.font = "20px Arial";
                    if (face.gender.toLowerCase()=="female"){
                        ctx.fillStyle = "#EC1EEF";
                        ctx.fillText('Female', face.x, face.y-4);
                        ctx.strokeStyle="#EC1EEF";
                    }else{
                        ctx.fillStyle = "#1EB5EF";
                        ctx.fillText('Male', face.x, face.y-4);
                        ctx.strokeStyle="#1EB5EF";
                    }

                    ctx.strokeRect(face.x, face.y, face.width, face.height);
                }
            }

            $('.loaderModel').hide(); // Hidet the loader
            $('#resultModel').fadeIn(600); // We make appear the div that will contain the prediction
            //$('#resultModel').text(data.meta_info); // We can show or no the message of the python result
            if (data.faces==undefined || Object.keys(data.faces).length<=0){
                $('#resultModel').text("No faces found!"); // If there were no faces, we notify
            }
        },
    });
}