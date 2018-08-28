let fake_predict = function(test_type) {
    if (test_type == 1) {
        var url = '/predictfakeERS'
    }
    else {
        var url = '/predictfakeCLASS'
    }
    $.ajax({
        url: url,
        contentType: "application/json; charset=utf-8",
        type: 'POST',
        success: function (data) {
            display_fake_prediction(data)
        }
    });s
};

let display_fake_prediction = function(prediction) {
    if (prediction.test_type == "ers") {
        $("img#prediction_ers").attr('src', '/static/images/demo/ers_graph.jpg')
    }
    else if (prediction.test_type == "class") {
        $("img#prediction_class").attr('src', '/static/images/demo/class_graph.jpg')
    }
};

let get_graph = function(test_type) {
    if (test_type == 1) {
        var url = '/predictERS'
    }
    else {
        var url = '/predictCLASS'
    }
    var data = $('form').serializeArray()
    $.ajax({
        data: JSON.stringify(data),
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
        $("img#prediction_ers").attr('src', '/static/images/prediction-graphs/ers_graph.jpg')
    }
    else if (prediction.test_type == "class") {
        $("img#prediction_class").attr('src', '/static/images/prediction-graphs/class_graph.jpg')
    }
};