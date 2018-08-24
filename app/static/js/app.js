let get_predict = function(test_type) {
    if (test_type == "ers") {
        var url = '/predictERS'
    }
    else {
        var url = '/predictCLASS'
    }
    $.ajax({
        url: url,
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_prediction(data)
        }
    });
};
let display_prediction = function(prediction) {
    if (prediction.test_type == "ers") {
        $("span#prediction_ers").html(prediction.prediction)
    }
    else if (prediction.test_type == "class") {
        $("span#prediction_class").html(prediction.prediction)
    }
    else {
        $("span#error").html("There was an error predicting a score")
    }
};