


var tableModule = (function() {



    function get_table_html(tid, tdata, tcaption) {

        var t_html = "";

        t_html = "<div id='table_area_common' class='col-md-12'>";
        t_html += "<div class='table_section mt-2 ml-2 table-responsive' id='table_section_" + tid  + "' >";
        t_html += "<div class='table_section_title mt-2 mr-2'>" + tid + "</div>";
        t_html += "<div class='float-left mt-2 mr-2 mb-2 table_caption_common'>*" +  tcaption + "</div>";
        //t_html += "<div class='table_section_fixed_head mr-2'></div>";
        t_html += "<div class='table_section_holder mr-2'>" + tdata + "</div>";
        t_html += "</div>"; // close table_section
        t_html += "</div>"; // close table_area_common

        return t_html;
    }

    function table_beautify(tid, holderElem) {

        // Add required classes for styles
        var curSectionElem = holderElem.find("#table_section_" + tid);
        var curSectionTableElem = curSectionElem.find('.dataframe');
        var curSectionTableHeadElem = curSectionTableElem.find('thead');
        
        // Add required classes
        curSectionTableElem.addClass('table text-center table-hover table-stripped table-sm');
        curSectionTableHeadElem.addClass('thead-dark');
        
        if(curSectionTableElem.height() > 400) {
            // Add scroll to only the table body
            // 400 is the max height we set to each table
            
            // Custom code
            //detachTableHead(curSectionElem);
            
            // Plugin - 1
            //curSectionTableElem.scrollTableBody()

            // Plugin - 2
            curSectionTableElem.floatThead({
                scrollContainer: function($table) {
                    return $table.closest('.table_section_holder');
                }
            });
        }
    }

    // When there is scroll to the table
    // This function detaches the head from table and creates a new table with only head
    // And remove head from original table
    // So that body is only scrollable
    function detachTableHead(holderElem) {
        var tableElem = holderElem.find('.table.dataframe');
        var headElem = tableElem.find('thead');
        var headHTML = headElem.html();

        var fixedHeadElem = holderElem.find('.table_section_fixed_head');
        
        // Erase previous content
        fixedHeadElem.html('');

        var fHeadHTML = "<table class='table table-sm table-dark mb-0'>";
        fHeadHTML += headHTML;
        fHeadHTML += "</table>"

        fixedHeadElem.append(fHeadHTML);

        fixedHeadElem.find("table tr th").each(function(i) {
            $(this).width($(tableElem.find("tr:first th")[i]).width());
        });

        headElem.remove();
    }

    // All the functions which are accessible outside
    return {
        'get_table_html': get_table_html, // HTML for the table component
        'table_beautify': table_beautify // Beautifying the table through Bootstrap classes
    }
  
})();