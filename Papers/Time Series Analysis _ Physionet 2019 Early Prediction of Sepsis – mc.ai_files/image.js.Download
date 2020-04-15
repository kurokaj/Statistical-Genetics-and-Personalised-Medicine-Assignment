jQuery(document).ready(function ($) {
    // lazy load
    if (fifuImageVars.fifu_lazy == 'on') {
        jQuery.extend(jQuery.lazyLoadXT, {
            srcAttr: 'data-src',
            visibleOnly: false,
            updateEvent: 'load orientationchange resize scroll touchmove focus hover'
        });
    }

    // woocommerce lightbox/zoom
    disableClick($);
    disableLink($);

    // for all images at single product page
    setTimeout(function () {
        resizeImg($);
        jQuery('a.woocommerce-product-gallery__trigger').css('visibility', 'visible');
    }, 2500);

    // zoomImg
    setTimeout(function () {
        jQuery('img.zoomImg').css('z-index', '');
    }, 1000);
});

jQuery(window).on('ajaxComplete', function () {
    if (fifuImageVars.fifu_lazy == 'on') {
        setTimeout(function () {
            jQuery(window).lazyLoadXT();
        }, 300);
    }
});

jQuery(window).on('load', function () {
    jQuery('.flex-viewport').css('height', '100%');
});

function resizeImg($) {
    var imgSelector = ".post img, .page img, .widget-content img, .product img, .wp-admin img, .tax-product_cat img, .fifu img";
    var resizeImage = function (sSel) {
        jQuery(sSel).each(function () {
            //original size
            var width = $(this)['0'].naturalWidth;
            var height = $(this)['0'].naturalHeight;

            //100%
            var ratio = width / height;
            jQuery(this).attr('data-large_image_width', jQuery(window).width() * ratio);
            jQuery(this).attr('data-large_image_height', jQuery(window).width());
        });
    };
    resizeImage(imgSelector);
}

function disableClick($) {
    if (!fifuImageVars.fifu_woo_lbox_enabled) {
        firstParentClass = '';
        parentClass = '';
        jQuery('figure.woocommerce-product-gallery__wrapper').find('div.woocommerce-product-gallery__image').each(function (index) {
            parentClass = jQuery(this).parent().attr('class').split(' ')[0];
            if (!firstParentClass)
                firstParentClass = parentClass;

            if (parentClass != firstParentClass)
                return false;

            jQuery(this).children().click(function () {
                return false;
            });
            jQuery(this).children().children().css("cursor", "default");
        });
    }
}

function disableLink($) {
    if (!fifuImageVars.fifu_woo_lbox_enabled) {
        firstParentClass = '';
        parentClass = '';
        jQuery('figure.woocommerce-product-gallery__wrapper').find('div.woocommerce-product-gallery__image').each(function (index) {
            parentClass = jQuery(this).parent().attr('class').split(' ')[0];
            if (!firstParentClass)
                firstParentClass = parentClass;

            if (parentClass != firstParentClass)
                return false;

            jQuery(this).children().attr("href", "");
        });
    }
}
