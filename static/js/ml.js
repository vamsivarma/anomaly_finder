$(document).ready(function() {
    register_ml_events();
});


function register_ml_events() {
    var mlElem = $("#ml-section-holder");

    mlElem.find(".dropdown-menu li a").click(function(e){
        e.preventDefault()
        var selText = $(this).text();
        $(this).parents('.btn-group').find('.dropdown-toggle').html(selText+' <span class="caret"></span>');
    });

    mlElem.find("form").submit(function(e){
        return false;
    });

    mlElem.find("#ml_submit").click(function(e) {
        e.preventDefault();
        ml_predict();
    });

}                       
                                  
function ml_predict() {
    var mlElem = $("#ml-section-holder");

    var ml_word = mlElem.find("#ml_word").val();
    var ml_model = mlElem.find("#ml_model").val();

    var api_data = {
        'word': ml_word,
        'model': ml_model
    }

    $.ajax({ 
        url: '/predict', 
        type: 'POST',
        data: JSON.stringify(api_data),
        dataType: "json", 
        success: function(response) { 
            render_ml_result(response);
        },
        error: function(response) {
            console.log("Predict API failed")
        } 
    }); 

}

// @TODO: Need to do better exception handling here
function render_ml_result(res) {

    // Erasing the previous content
    var mlResElem = $("#ml-section-holder .ml-results");
    mlResElem.html("");

    // @TODO: Do proper exception handling here...
    if(res) {
        mlResElem.append(getMLResHTML(res));
    } else {
        // @TODO: Display empty results message here...
    }
}

function getMLResHTML(ml_res) {
    var mlResHTML = "<h3> Prediction results: </h3>";
    mlResHTML += "<h6 class='result'> Francais: " +  ml_res['fr'] + "%</h6>";
    mlResHTML += "<h6 class='result'> English: " +  ml_res['en'] + "%</h6>";
    mlResHTML += "<h6 class='result'> Espanol: " +  ml_res['es'] + "%</h6>";
    return mlResHTML;
}


                                  
                                  