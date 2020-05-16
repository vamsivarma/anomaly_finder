$(window).on('load', function() {
    exploreModule.initModule();
});

var exploreModule = (function() {

    var _this = '';
    var eElem = '';
    var drpElem = '';

    var caption_meta = {
        "numeric": "Summary statistics of all the numeric fields of the data set",
        "categorical": "Summary statistics of all the categorical fields of the data set",
        "properties": "Structure of current dataset"
    }

    var summary_order = ["properties", "numeric", "categorical"];

    function initModule() {

        _this = this;
        eElem = $("#explore-area");
        drpElem = eElem.find("#eDSList");
        btnElem = eElem.find("#explore_ds");
        eSElem = eElem.find("#explore_ds_summary");

        register_module_events();
    }

    function renderModule() {
        // Only for the first time load
        if(drpElem.html() == '') {
            render_datasets();
        }
    }

    function render_datasets() {

        var row_data = commonModule.get_datasets();

        // Erasing the previous content
        drpElem.html("");

        var rowCount = row_data.length;

        if(rowCount) {

            for(var i=0; i< rowCount; i++) {

                ds_name = row_data[i]['name'];
                
                // Add a separate option in the dropdown for each dataset in the response
                drpElem.append(commonModule.getOptionHTML(ds_name));
            }
        } else {
            // @TODO: Display empty data message
        }
    }


    function register_module_events() {

        // Dataset explore button click
        btnElem.click(function(e) {

            var api_data = {
                'd_name': drpElem.val()
            }  

            apiModule.explore_dataset(api_data, dataset_summary.bind(this));

        });

    }


    function dataset_summary(ds_summary) {

        var eSElem = eElem.find("#explore_ds_summary");
        // Clearing the previous content
        eSElem.html('');

        var nativeESElem = document.getElementById('explore_ds_summary');

        for(i in summary_order) {
            t = summary_order[i];

            var t_html = tableModule.get_table_html(t, ds_summary[t], caption_meta[t]);
            nativeESElem.insertAdjacentHTML('beforeend', t_html );

            tableModule.table_beautify(t, eSElem);
        }

        //eSElem.find('.dataframe.table').scrollTableBody();
    }

    return {
        'initModule': initModule,
        'renderModule': renderModule
    }
  
})();