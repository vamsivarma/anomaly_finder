$(window).on('load', function() {
    experimentsModule.initModule();
});

var experimentsModule = (function() {

    var _this = '';
    
    // Holder elem for experiments section
    var expElem = '';
    
    // Datasets dropdown
    var expDSElem = '';
    
    // Target label dropdown
    var expLabelElem = '';

    // Button for initiating experiment
    var expBtnElem = '';

    // Experiment results holder
    var expResultsElem = '';

    // Importance - for finding feature importance
    // Metrics - for finding the matrics for regular machine learning algorithms
    var exp_list = ['importance', 'metrics']; // , 'field', 'feature'];

    var caption_meta = {
        "importance": "Features sorted by their importance",
        "metrics": "Metric values sorted by accuracy"
    }

    var ml_alogirithm_map = {
        "SVM": "Support Vector Machines", 
        "PCA": "Principle Component Analysis", 
        "DT": "Decision Trees", 
        "RF": "Random Forest", 
        "NB": "Naive Bayes", 
        "SGD": "Stocastic Gradient Descent", 
        "XB": "XGBoost", 
        "LR": "Logistic Regression", 
        "KNN": "K Nearest Neighbours"
    }
    
    function initModule() {

        _this = this;
        expElem = $("#experiments-area");
        expDSElem = expElem.find("#expDSList");
        expLabelElem = expElem.find("#expDSLabel");
        expBtnElem = expElem.find("#expDSLaunch");
        expResultsElem = expElem.find("#expResultsHolder");
        expSplit = expElem.find("#expSplit");

        register_module_events();
    }


    function renderModule() {

        //Get the current datasets
        render_datasets();
        
    }

    function render_datasets() {

        var row_data = commonModule.get_datasets();

        // Erasing the previous content
        expDSElem.html("");

        var rowCount = row_data.length;

        if(rowCount) {

            for(var i=0; i< rowCount; i++) {

                var cur_ds_name = row_data[i]['name'];
                
                // Add a separate option in the dropdown for each dataset in the response
                expDSElem.append(commonModule.getOptionHTML(cur_ds_name));
            }

        } else {

            // @TODO: Display empty data message
        
        }
    }

    function register_module_events() {

        // Datasets dropdown change event
        // Here for every dataset we show only categorical variables
        expDSElem.on('change', function() {
            var curDS = $(this).val();
            var datasets_meta = commonModule.get_ds_meta();
            var curDSColMap = datasets_meta[curDS]['col_type_map'];
            
            commonModule.render_dropdown(commonModule.get_cat_cols(curDSColMap), expLabelElem);
        });

        // Experiment button click
        expBtnElem.click(function(e) {

            var api_data = {
                'e_list':  exp_list,//aElem.find('#chart_select').val(),
                'd_name': expDSElem.val(),
                'label_col': expLabelElem.val(),
                'd_split': parseFloat(expSplit.val())
            }  

            apiModule.launch_experiment(api_data, render_experiment.bind(this), '');

        });

    }

    // @TODO: Need to do exception handling properly
    function render_experiment(e_data) {

        var expResultsElem = expElem.find("#expResultsHolder");
        
        // Clearing the previous content
        expResultsElem.html('');

        var nativeESElem = document.getElementById('expResultsHolder');

        for(i in exp_list) {
            t = exp_list[i];

            var t_html = tableModule.get_table_html(t, e_data[t], caption_meta[t]);
            nativeESElem.insertAdjacentHTML('beforeend', t_html );

            tableModule.table_beautify(t, expResultsElem);
        }

    }
    
    return {
        'initModule': initModule,
        'renderModule': renderModule
    }
  
})();


                                  



                                  
                                  