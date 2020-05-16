$(window).on('load', function() {
    settingsModule.initModule()
});

var settingsModule = (function() {

    var sHolderElem = '';

    // For saving the settings to Mongo DB
    var settings_obj = {}

    // Differentiating the type of elements in the form
    var setting_type = {
        "checkbox": ['use_gpu', 'normalize', 'save_log', 'export_db', 
                    'export_csv', 'dataset_view', 'history_view', 
                    'verbose', 'save_best_model', 'early_stopping' ],
        "input": ['n_hidden', 'epochs_number', 'learning_rate', 'train_batch_size', 'test_batch_size'], //'t'
        "select": ['gpu_type', 'input_columns', 'output_column', 'model_type', 'train_algorithm', 'neural_model']
    }

    // Setting name to DOM ID mapping
    var db_map = {
        "use_gpu": "cUseGPU",
        "gpu_type": "cGPUType",
        //"t": "cT",
        "normalize": "cN",
        "input_columns": "cInputCols",
        "output_column": "cOutputCols",
        "save_log": "cSaveLog",
        "export_db": "cSaveDB",
        "export_csv": "cSaveCSV",
        "dataset_view": "cDatasetV",
        "history_view": "cHistoryV",
        "verbose": "cVerbose",
        "model_type": "cModelType",
        "n_hidden": "cHC",
        "train_algorithm": "cTA",
        "epochs_number": "cEpochsCount",
        "learning_rate": "cLR",
        "save_best_model": "cSBM",
        "early_stopping": "cES",
        "train_batch_size": "cTrainBS",
        "test_batch_size": "cTestBS",
        "neural_model": "cNeuralModel"
    }

    function initModule() {

        sHolderElem = $("#settings-area");

        register_settings_events();
    }

    function register_settings_events() {

        // @TODO: Need to do this more efficiently
        sHolderElem.find('.card-header').click(function(e) {
            
            sHolderElem.find('.cplus').removeClass('d-none').addClass('d-block');
            sHolderElem.find('.cminus').removeClass('d-block').addClass('d-none');
    
            $(this).find('.commons').removeClass('d-block').addClass('d-none');
    
            if(!$(this).hasClass('collapsed')) {
                $(this).find('.cplus').addClass('d-block');
            } else {
                $(this).find('.cminus').addClass('d-block');
            }
        });
    }

    function renderModule() {

        //Render only for the first time load
        if($.isEmptyObject(settings_obj)) {
            //Get the current settings
            get_global_settings();
        }
    }

    function get_global_settings() {

        var api_data = {
            'user_name': "fabio",
            'user_id': 111
        }
    
        apiModule.get_settings(api_data, fill_ad_settings.bind(this))
    }
    

    function fill_ad_settings(response) {
        var ad_setting = response['sconfig'];
        settings_obj = ad_setting;
    
        //Set the settings as per the values from API
        settings_binder(ad_setting, true);
    }
    
    
    
    // if sflag is true then set the settings from API
    // if sflag is false then get the settings to be saved in DB via API
    function settings_binder(s_config, sflag) {
    
        for(key in setting_type) {
    
            var type_elems = setting_type[key];
    
            if(key == "checkbox") {
                type_elems.forEach(function(elem) { 
    
                    //console.log("Checkbox: " + elem);
                    var domID = db_map[elem];
                    if(sflag) {
                        sHolderElem.find("#" + domID).prop('checked', s_config[elem]);
                    } else {
                        s_config[elem] = sHolderElem.find("#" + domID).prop('checked');
                    }
                    
                });
    
            } else if(key == "input" || key == "select") {
    
                type_elems.forEach(function(elem) { 
    
                    //console.log("Input or Select element: " + elem);
                    var domID = db_map[elem];
                    if(sflag) {
                        sHolderElem.find("#" + domID).val(s_config[elem]);
                    } else {
                        s_config[elem] = sHolderElem.find("#" + domID).val();
                    }
    
                });
    
            } else {
                // Setting remaning elements comes here... 
            }
    
        }
    
        if(!sflag) {
            return s_config;
        }
    }
    
    function save_settings() {
    
        //Get the settings as per the user selections
        var settings = settings_binder(settings_obj, false);
    
        var api_data = {
            'user_name': "fabio",
            'user_id': 111,
            's_config': settings 
        }

        apiModule.save_settings(api_data, show_save_success.bind(this))
    
    }
    
    function show_save_success(res) {
        //alert("Settings are saved successfully");
    }
    
    return {
        'initModule': initModule,
        'renderModule': renderModule,
        'save_settings': save_settings
    }
    
})();




