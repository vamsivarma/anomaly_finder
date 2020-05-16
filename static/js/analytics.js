$(window).on('load', function() {
    analyticsModule.initModule();
});

var analyticsModule = (function() {

    var _this = '';
    var aElem = '';
    var datasetDrpElem = '';
    var drpElem = '';

    var charts_meta = {};

    // This object is to show or hide the drop down for a particular chart
    var chart_drp_meta = {
        'heatmap': false,
        'histogram': true,
        'box': true,
        'field': false,
        'pie': true,
        'feature': false
    }

    var charts_list = ['pie', 'box', 'histogram', 'heatmap']; // , 'field', 'feature'];
    
    function initModule() {

        _this = this;
        aElem = $("#analytics-area");
        datasetDrpElem = aElem.find("#dsList");
        drpElem = aElem.find("#dsLabel");

        register_module_events();
    }


    function renderModule() {

        // @TODO: Need to cover the case of new datasets being uploaded 
        // After initial page load
        if($.isEmptyObject(charts_meta)) {
            //Get the current datasets
            render_datasets();
        }
        
    }

    function render_datasets() {

        var row_data = commonModule.get_datasets();

        // Erasing the previous content
        datasetDrpElem.html("");

        var rowCount = row_data.length;

        if(rowCount) {

            // Initializing target dropdown
            
            // If the datasets exist then get the fields of the first dataset to display
            // Target column values
            commonModule.render_dropdown(commonModule.get_cat_cols(row_data[0]['col_type_map']), drpElem);

            for(var i=0; i< rowCount; i++) {

                ds_name = row_data[i]['name'];
                
                // Add a separate option in the dropdown for each dataset in the response
                datasetDrpElem.append(commonModule.getOptionHTML(ds_name));
            }


        } else {
            // @TODO: Display empty data message
        }
    }

    

    function register_module_events() {

        // Datasets dropdown change event
        // Here for every dataset we show only categorical variables
        aElem.find('#dsList').on('change', function() {
            var curDS = $(this).val();
            var datasets_meta = commonModule.get_ds_meta();
            var curDSColMap = datasets_meta[curDS]['col_type_map'];
            var drpElem = aElem.find("#dsLabel");

            commonModule.render_dropdown(commonModule.get_cat_cols(curDSColMap), drpElem);
        });

        // Dataset analyze button click
        aElem.find('#analyze_ds').click(function(e) {

            var api_data = {
                'charts_list':  charts_list,//aElem.find('#chart_select').val(),
                'd_name': aElem.find('#dsList').val(),
                'label_col': aElem.find('#dsLabel').val()
            }  

            apiModule.get_analytics(api_data, render_charts.bind(this), '');

        });

    }

    // @TODO: Need to do exception handling properly
    function render_charts(charts_data) {

        var chartHolderElem = $("#charts_section_holder");
        var nativeHElem = document.getElementById('charts_section_holder');
        var curDS = $("#dsList").val();
        //var curLabel = $("#dsLabel").val();
        //var curDSCols = datasets_meta[curDS]['columns'];
        
        var datasets_meta = commonModule.get_ds_meta();
        var curDSColTypes = datasets_meta[curDS]['col_type_map'];
        
        charts_meta = {};

        // Erasing the previous data
        chartHolderElem.html("");

        for(chart in charts_data) {
            
            var chartHTML = get_chart_html(chart, chart_drp_meta[chart]);
            nativeHElem.insertAdjacentHTML('beforeend', chartHTML );
            //charts_section_holder.append(chartHTML);
            //console.log(charts_data[chart]);
            charts_meta[chart] = charts_data[chart];
            Plotly.newPlot( chart + '_chart', JSON.parse(charts_meta[chart]['data']) );

            // Render the dropdown for the chart
            if(chart_drp_meta[chart]) {
                var curChartDrpElem = chartHolderElem.find("#ds" + chart + "ChartDrp");
                var curChartCols = [];
                var curChartTypes = charts_meta[chart]['chart_types'];
                
                for(col in curDSColTypes) {
                    if(curChartTypes.includes(curDSColTypes[col])) {
                        curChartCols.push(col);
                    }
                }

                commonModule.render_dropdown(curChartCols, curChartDrpElem);
                curChartDrpElem.val(charts_meta[chart]['label_col']);
            }
        }

        register_chart_events();

    }

    function get_chart_html(chart_name, showDrpFlag) {

        var chart_id = chart_name + "_chart";

        var chartHTML = "<div class='chart-card-holder col-md-6 p-2'>"; //Div 1 open

        if(showDrpFlag) {
            chartHTML += "<div id='pageSpinnerHolder' class='inlineSpinnerHolder'>";
            chartHTML += "<div class='pageSpinner'>";
            chartHTML += "<img src='/static/images/loader.gif' class='pageSpinnerImage'>";
            chartHTML += "<div class='pageSpinnerText'>Loading...</div>";
            chartHTML += "</div></div>";
        }

        chartHTML += "<div class='card card-common' rel='" + chart_name + "' >"; //Div 2 open
        chartHTML += "<div class='card-body'>"; //Div 3 open
        chartHTML += "<div class='chart-card-title'>" + chart_name + "</div>"; //Div 4 open and close

        if(showDrpFlag) {
            chartHTML += "<div class='chart-dropdown-holder mt-5'>";
            chartHTML += "<select class='form-control dsChartDrp' id='ds" + chart_name + "ChartDrp'>";
            chartHTML += "</select></div>";
        }

        chartHTML += "<div class='d-flex justify-content-between'>"; //Div 5 open
        chartHTML += "<div class='chart_holder_common' id='" + chart_id + "' >"; //Div 6 open
        //chartHTML += "<script>" + Plotly.plot(chart_id, chart_data,{}) + "</script>"; // Script open and close
        chartHTML += "</div>"; // Div 6 close
        chartHTML += "</div>"; // Div 5 close
        chartHTML += "</div>"; // Div 3 close
        chartHTML += "<div class='card-footer text-secondary'><div class='chart-card-expand'>"; // Div 7,8 open
        chartHTML += "<i class='fas fa-expand float-right'></i>"; // i open and close 
        chartHTML += "</div></div>"; // Div 7,8 close
        chartHTML += "</div>"; // Div 2 close
        chartHTML += "</div>"; // Div 1 close

        // This is a way to "htmlDecode" your string...    
        //chartHTML = $("<div />").html(chartHTML).html();
        //chartHTML = $.parseHTML(chartHTML) 

        return chartHTML;

    }

    function register_chart_events() {
        var aElem = $("#analytics-area");

        aElem.find(".chart-card-expand").click(function(e) {
            
            e.preventDefault();

            var cardHolderElem = $(this).closest('.card.card-common');
            var chartTitle = cardHolderElem.find('.chart-card-title').text();
            //var chartType = cardHolderElem.prop('rel');
            var chartData = JSON.parse(charts_meta[chartTitle]['data']);
            var maxModalHolder = aElem.find('#analytics-chart-max');

            
            var modelDims = {
                'w': maxModalHolder.width(),
                'h': maxModalHolder.height()
            }

            // Setting the model window dimensions based on available screen height and width
            aElem.find("#max-chart-holder").css({
                'width': modelDims['w'] * 0.75, // Only 80% of available width
                'height': modelDims['h'] 
            });

            //Display the chart on enlarged window
            maxModalHolder.find('.modal-title').html(chartTitle);


            Plotly.newPlot( 'max-chart-holder', chartData );
            //maxModalHolder.find('.max-chart-image').prop('src', chartIMGSource);
            maxModalHolder.modal('show');

        });


        aElem.find('.dsChartDrp').on('change', function() {
            var curElem = $(this);
            var curField = curElem.val();

            // Get the current chart name
            var curChart = curElem.closest('.card-common').attr('rel');

            var spinnerElem = curElem.closest('.chart-card-holder').find('.inlineSpinnerHolder');

            var api_data = {
                'charts_list': [curChart],
                'd_name': aElem.find('#dsList').val(),
                'label_col': curField
            }  


            apiModule.get_analytics(api_data, chart_rerender.bind(this), spinnerElem);

        });
    }

    function chart_rerender(charts_data) {

        // Rerender the current chart based on the selected field
        for(chart in charts_data) {
            charts_meta[chart] = charts_data[chart]; //JSON.parse(charts_data[chart]);
            Plotly.newPlot( chart + '_chart', JSON.parse(charts_meta[chart]['data']) );
        }
    }
    
    
    return {
        'initModule': initModule,
        'renderModule': renderModule
    }
  
})();


                                  



                                  
                                  