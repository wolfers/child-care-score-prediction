let get_predict = function(test_type) {
    if (test_type == 1) {
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
        $("img#prediction_ers").attr('src', '/static/images/ers_graph.jpg')
    }
    else if (prediction.test_type == "class") {
        $("img#prediction_class").attr('src', '/static/images/class_graph.jpg')
    }
};