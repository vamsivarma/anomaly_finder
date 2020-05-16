$(window).on('load', function() {
    apiModule.initModule()
});


// All the API calls to the server are invoked through this module
var apiModule = (function() {

    var _this = '';
    var spinnerElem = '';

    function initModule() {
        _this = this;
        spinnerElem = $('.pageSpinnerHolder');
    }

    // @TODO: Need to make this as a common function
    function get_datasets(callbackFunc) {

        // Show the spinner before API call
        spinnerElem.show();
    
        $.ajax({ 
        url: '/datasets', 
        type: 'POST',
        dataType: "json", 
        success: function(response) { 
            // Hide the spinner on API call success or failure
            spinnerElem.hide();
            
            callbackFunc(response);
    
        },
        error: function(response) {
            // Hide the spinner on API call success or failure
            spinnerElem.hide();
            
            // @TODO: Need to use Bootstrap popover here
            alert("Datasets API failed")
        } 
        }); 
    
    }

    function toggleSpinner(sElem, sFlag) {

        if(sElem == '') {
            // Showing/Hiding full spinner
            if(sFlag) {
                spinnerElem.show();
            } else {
                spinnerElem.hide();
            }
        } else {
            // Showing/Hiding inline spinner
            if(sFlag) {
                sElem.show();
            } else {
                sElem.hide();
            }
        }

    }

    function get_analytics(api_data, callbackFunc, sElem) {
        
        // Show the spinner
        toggleSpinner(sElem, true)
        
        $.ajax({
            url: "/get_analytics",
            type: "POST",
            data: JSON.stringify(api_data),
            dataType: "json",
            success: function (response) {

                // Hide the spinner after API call
                toggleSpinner(sElem, false);

                callbackFunc(response);
            },
            error: function(response) {

                // Hide the spinner after API call
                toggleSpinner(sElem, false);
                
                alert("Analytics API failed")
            }
        });
    }

    function get_settings(api_data, callbackFunc) {

        // Show the spinner before API call
        spinnerElem.show();

        $.ajax({ 
            url: '/get_settings', 
            type: 'POST',
            data: JSON.stringify(api_data),
            dataType: "json", 
            success: function(response) { 
                // Hide the spinner after API call
                spinnerElem.hide();
    
                callbackFunc(response);
            },
            error: function(response) {
    
                // Hide the spinner after API call
                spinnerElem.hide();
    
                alert("GET Settings API failed")
            } 
        }); 
    }

    function save_settings(api_data, callbackFunc) {

        // Show the spinner before API call
        spinnerElem.show();

        $.ajax({ 
            url: '/save_settings', 
            type: 'POST',
            data: JSON.stringify(api_data),
            dataType: "json", 
            success: function(response) { 
    
                // Hide the spinner after API call
                spinnerElem.hide();
                
                callbackFunc(response);
            },
            error: function(response) {
    
                // Hide the spinner after API call
                spinnerElem.hide();
    
                alert("Save Settings API failed");
            } 
        }); 
    }

    function explore_dataset(api_data, callbackFunc) {

        // Show the spinner before API call
        spinnerElem.show();

        $.ajax({ 
            url: '/explore_dataset', 
            type: 'POST',
            data: JSON.stringify(api_data),
            dataType: "json", 
            success: function(response) { 
    
                // Hide the spinner after API call
                spinnerElem.hide();
                
                callbackFunc(response);
            },
            error: function(response) {
    
                // Hide the spinner after API call
                spinnerElem.hide();
    
                alert("Explore dataset API failed");
            } 
        }); 
    }

    function launch_experiment(api_data, callbackFunc, sElem) {
        
        // Show the spinner
        toggleSpinner(sElem, true)
        
        $.ajax({
            url: "/launch_experiment",
            type: "POST",
            data: JSON.stringify(api_data),
            dataType: "json",
            success: function (response) {

                // Hide the spinner after API call
                toggleSpinner(sElem, false);

                callbackFunc(response);
            },
            error: function(response) {

                // Hide the spinner after API call
                toggleSpinner(sElem, false);
                
                alert("Experiments API failed")
            }
        });
    }


    // @TODO: Need to get rid of all these functions and
    // maintain a single api function with metadata passed from outside
    return {
        'initModule': initModule,
        'get_datasets': get_datasets, // API for datasets
        'get_analytics': get_analytics, // API for charts in the Analytics section
        'get_settings': get_settings, // Get settings from Mongo
        'save_settings': save_settings, // Save settings to Mongo
        'explore_dataset': explore_dataset, // Summary statistics of each dataset
        'launch_experiment': launch_experiment // Initiate ml for current dataset
    }
  
})();