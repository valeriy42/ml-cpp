/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0 and the following additional limitation. Functionality enabled by the
 * files subject to the Elastic License 2.0 may only be used in production when
 * invoked by an Elasticsearch process with a license key installed that permits
 * use of machine learning features. You may not use this file except in
 * compliance with the Elastic License 2.0 and the foregoing additional
 * limitation.
 */

#include <core/CCTimeR.h>
#include <core/CLogger.h>
#include <core/CTimeUtils.h>
#include <core/CTimezone.h>

#include <boost/test/unit_test.hpp>

#include <chrono>
#include <ctime>
#include <thread>

BOOST_AUTO_TEST_SUITE(CTimeUtilsTest)

BOOST_AUTO_TEST_CASE(testNow) {
    ml::core_t::TTime t1(ml::core::CTimeUtils::now());
    std::int64_t t1Ms(ml::core::CTimeUtils::nowMs());
    std::this_thread::sleep_for(std::chrono::milliseconds(1001));
    ml::core_t::TTime t2(ml::core::CTimeUtils::now());
    std::int64_t t2Ms(ml::core::CTimeUtils::nowMs());

    BOOST_TEST_REQUIRE(t2 > t1);
    BOOST_TEST_REQUIRE(t2Ms > t1Ms);
}

BOOST_AUTO_TEST_CASE(testToIso8601) {
    // These tests assume UK time.  In case they're ever run outside the UK,
    // we'll explicitly set the timezone for the purpose of these tests.
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        ml::core_t::TTime t(1227710437);
        std::string expected("2008-11-26T14:40:37+0000");

        const std::string strRep = ml::core::CTimeUtils::toIso8601(t);

        BOOST_REQUIRE_EQUAL(expected, strRep);
    }
    {
        ml::core_t::TTime t(1207925624);
        std::string expected("2008-04-11T15:53:44+0100");

        const std::string strRep = ml::core::CTimeUtils::toIso8601(t);

        BOOST_REQUIRE_EQUAL(expected, strRep);
    }
}

BOOST_AUTO_TEST_CASE(testToLocal) {
    // These tests assume UK time.  In case they're ever run outside the UK,
    // we'll explicitly set the timezone for the purpose of these tests.
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        ml::core_t::TTime t(1227710437);
        std::string expected("Wed Nov 26 14:40:37 2008");

        const std::string strRep = ml::core::CTimeUtils::toLocalString(t);

        BOOST_REQUIRE_EQUAL(expected, strRep);
    }
    {
        ml::core_t::TTime t(1207925624);
        std::string expected("Fri Apr 11 15:53:44 2008");

        const std::string strRep = ml::core::CTimeUtils::toLocalString(t);

        BOOST_REQUIRE_EQUAL(expected, strRep);
    }
    {
        ml::core_t::TTime t(1207925624);
        std::string expected("15:53:44");

        const std::string strRep = ml::core::CTimeUtils::toTimeString(t);

        BOOST_REQUIRE_EQUAL(expected, strRep);
    }
}

BOOST_AUTO_TEST_CASE(testToEpochMs) {
    BOOST_REQUIRE_EQUAL(1000, ml::core::CTimeUtils::toEpochMs(ml::core_t::TTime(1)));
    BOOST_REQUIRE_EQUAL(-1000, ml::core::CTimeUtils::toEpochMs(ml::core_t::TTime(-1)));
    BOOST_REQUIRE_EQUAL(1521035866000,
                        ml::core::CTimeUtils::toEpochMs(ml::core_t::TTime(1521035866)));
    BOOST_REQUIRE_EQUAL(-1521035866000,
                        ml::core::CTimeUtils::toEpochMs(ml::core_t::TTime(-1521035866)));
}

BOOST_AUTO_TEST_CASE(testStrptime) {
    // These tests assume UK time.  In case they're ever run outside the UK,
    // we'll explicitly set the timezone for the purpose of these tests.
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Europe/London"));

    {
        // This time is deliberately chosen to be during daylight saving time
        std::string dateTime("1122334455");

        std::string format("%s");
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
#ifndef Windows
        // This fails on Windows unless the operating system timezone is set to UK time.
        // This means that using %s as a time format doesn't work on Windows.  The reason
        // is that the underlying strptime() returns a struct tm, so the seemingly most
        // simple conversion gets round-tripped through an intermediate step that relies
        // on timezone functionality.  A fix would be non-trivial, and since this problem
        // doesn't affect production code it's not worth the effort.  In the production
        // code all date parsing is done in the Java code.  Date parsing is only used in
        // the C++ code when running a program for test/debug purposes with the
        // --timeformat option.  Generally we'd be doing this on macOS or Linux, but even
        // if someone did want to do testing/debugging on Windows by simply not specifying
        // the --timeformat option the time is assumed to be in epoch format and converted
        // by a simple string to number conversion rather than using strptime().  So it
        // really would be a waste of effort getting %s to work on Windows at this time.
        ml::core_t::TTime expected(1122334455);
        BOOST_REQUIRE_EQUAL(expected, actual);
#endif
    }
    {
        std::string dateTime("2008-11-26 14:40:37");

        std::string format("%Y-%m-%d %H:%M:%S");

        ml::core_t::TTime expected(1227710437);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        std::string badDateTime("2008-11-26 25:40:37");
        BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::strptime(format, badDateTime, actual));
    }
    {
        std::string dateTime("10/31/2008 3:15:00 AM");

        std::string format("%m/%d/%Y %I:%M:%S %p");

        ml::core_t::TTime expected(1225422900);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);
        LOG_DEBUG(<< actual);
    }
    {
        std::string dateTime("Fri Oct 31  3:15:00 AM GMT 08");

        std::string format("%a %b %d %I:%M:%S %p %Z %y");

        ml::core_t::TTime expected(1225422900);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);
        LOG_DEBUG(<< actual);
    }
    {
        std::string dateTime("Tue Jun 23  17:24:55 2009");

        std::string format("%a %b %d %T %Y");

        ml::core_t::TTime expected(1245774295);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);
        LOG_DEBUG(<< actual);
    }
    {
        std::string dateTime("Tue Jun 23  17:24:55 BST 2009");

        std::string format("%a %b %d %T %Z %Y");

        ml::core_t::TTime expected(1245774295);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);
        LOG_DEBUG(<< actual);
    }
    {
        // This time is in summer, but explicitly specifies a GMT offset of 0,
        // so we should get 1245777895 instead of 1245774295
        std::string dateTime("Tue Jun 23  17:24:55 2009 +0000");

        std::string format("%a %b %d %T %Y %z");

        ml::core_t::TTime expected(1245777895);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);
        LOG_DEBUG(<< actual);

        std::string badDateTime1("Tue Jun 23  17:24:55 2009");
        BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::strptime(format, badDateTime1, actual));

        std::string badDateTime2("Tue Jun 23  17:24:55 2009 0000");
        BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::strptime(format, badDateTime2, actual));
    }
    {
        // Test what happens when no year is given
        std::string dateTime("Jun 23  17:24:55");

        std::string format("%b %d %T");

        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        LOG_DEBUG(<< actual);

        // This test is only approximate (assuming leap year with leap second), so
        // print a warning too
        BOOST_TEST_REQUIRE(actual >= ml::core::CTimeUtils::now() - 366 * 24 * 60 * 60 - 1);
        char buf[128] = {'\0'};
        LOG_WARN(<< "If the following date is not within the last year then something is wrong: "
                 << ml::core::CCTimeR::cTimeR(&actual, buf));

        // Allow small tolerance in case of clock discrepancies between machines
        BOOST_TEST_REQUIRE(actual <= ml::core::CTimeUtils::now() +
                                         ml::core::CTimeUtils::MAX_CLOCK_DISCREPANCY);
    }
    {
        // Test what happens when no year is given
        std::string dateTime("Jan 01  01:24:55");

        std::string format("%b %d %T");

        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        LOG_DEBUG(<< actual);

        // This test is only approximate (assuming leap year with leap second), so
        // print a warning too
        BOOST_TEST_REQUIRE(actual >= ml::core::CTimeUtils::now() - 366 * 24 * 60 * 60 - 1);
        char buf[128] = {'\0'};
        LOG_WARN(<< "If the following date is not within the last year then something is wrong: "
                 << ml::core::CCTimeR::cTimeR(&actual, buf));

        // Allow small tolerance in case of clock discrepancies between machines
        BOOST_TEST_REQUIRE(actual <= ml::core::CTimeUtils::now() +
                                         ml::core::CTimeUtils::MAX_CLOCK_DISCREPANCY);
    }
    {
        // Test what happens when no year is given
        std::string dateTime("Dec 31  23:24:55");

        std::string format("%b %d %T");

        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        LOG_DEBUG(<< actual);

        // This test is only approximate (assuming leap year with leap second), so
        // print a warning too
        BOOST_TEST_REQUIRE(actual >= ml::core::CTimeUtils::now() - 366 * 24 * 60 * 60 - 1);
        char buf[128] = {'\0'};
        LOG_WARN(<< "If the following date is not within the last year then something is wrong: "
                 << ml::core::CCTimeR::cTimeR(&actual, buf));

        // Allow small tolerance in case of clock discrepancies between machines
        BOOST_TEST_REQUIRE(actual <= ml::core::CTimeUtils::now() +
                                         ml::core::CTimeUtils::MAX_CLOCK_DISCREPANCY);
    }
}

BOOST_AUTO_TEST_CASE(testTimezone) {
    static const ml::core_t::TTime SECONDS_PER_HOUR = 3600;

    // These convert the same date/time to a Unix time, but in a variety of
    // different timezones.  Since Unix times represent seconds since the epoch
    // UTC, the timezone will change the results.

    std::string format("%Y-%m-%d %H:%M:%S");
    std::string dateTime("2008-11-26 14:40:37");

    // Additionally, for each timezone, we'll try converting the same time,
    // but with UTC explicitly specified.  This should always come up with
    // the utcExpected time.  Also, to exercise the time convertor, we'll
    // explicitly specify 2 hours behind GMT (although it's unlikely this
    // would ever occur in a real log file).

    std::string formatExplicit("%Y-%m-%d %H:%M:%S %z");

    std::string dateTimeUtc("2008-11-26 14:40:37 +0000");
    ml::core_t::TTime utcExpected(1227710437);

    std::string dateTimeTwoHoursBehindUtc("2008-11-26 14:40:37 -0200");
    ml::core_t::TTime twoHoursBehindUtc(utcExpected + 2 * SECONDS_PER_HOUR);

    // UK first
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Europe/London"));
    {
        ml::core_t::TTime expected(utcExpected);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(formatExplicit, dateTimeUtc, actual));
        BOOST_REQUIRE_EQUAL(utcExpected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
            formatExplicit, dateTimeTwoHoursBehindUtc, actual));
        BOOST_REQUIRE_EQUAL(twoHoursBehindUtc, actual);
    }

    // US eastern time: 5 hours behind the UK (except during daylight saving
    // time switchover)
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("America/New_York"));
    {
        // The Unix time is in UTC, and UTC will be 5 hours ahead of US eastern
        // time at this time of the year (UTC is only 4 hours ahead in summer).
        ml::core_t::TTime expected(utcExpected + 5 * SECONDS_PER_HOUR);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(formatExplicit, dateTimeUtc, actual));
        BOOST_REQUIRE_EQUAL(utcExpected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
            formatExplicit, dateTimeTwoHoursBehindUtc, actual));
        BOOST_REQUIRE_EQUAL(twoHoursBehindUtc, actual);
    }

    // US Pacific time: 8 hours behind the UK (except during daylight saving
    // time switchover)
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("America/Los_Angeles"));
    {
        ml::core_t::TTime expected(utcExpected + 8 * SECONDS_PER_HOUR);
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(formatExplicit, dateTimeUtc, actual));
        BOOST_REQUIRE_EQUAL(utcExpected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
            formatExplicit, dateTimeTwoHoursBehindUtc, actual));
        BOOST_REQUIRE_EQUAL(twoHoursBehindUtc, actual);
    }

    // Australian central time: 9.5 hours ahead of GMT all year around in the
    // Northern Territory; in South Australia, 9.5 hours ahead of GMT in the
    // (southern hemisphere) winter and 10.5 hours ahead of GMT in the (southern
    // hemisphere) summer.

    // Northern Territory first
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Australia/Darwin"));
    {
        ml::core_t::TTime expected(
            utcExpected - static_cast<ml::core_t::TTime>(9.5 * SECONDS_PER_HOUR));
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(formatExplicit, dateTimeUtc, actual));
        BOOST_REQUIRE_EQUAL(utcExpected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
            formatExplicit, dateTimeTwoHoursBehindUtc, actual));
        BOOST_REQUIRE_EQUAL(twoHoursBehindUtc, actual);
    }

    // Now South Australia - remember, 26th November is summer in Australia,
    // so daylight saving is in force
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone("Australia/Adelaide"));
    {
        ml::core_t::TTime expected(
            utcExpected - static_cast<ml::core_t::TTime>(10.5 * SECONDS_PER_HOUR));
        ml::core_t::TTime actual(0);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(format, dateTime, actual));
        BOOST_REQUIRE_EQUAL(expected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(formatExplicit, dateTimeUtc, actual));
        BOOST_REQUIRE_EQUAL(utcExpected, actual);

        BOOST_TEST_REQUIRE(ml::core::CTimeUtils::strptime(
            formatExplicit, dateTimeTwoHoursBehindUtc, actual));
        BOOST_REQUIRE_EQUAL(twoHoursBehindUtc, actual);
    }

    // Set the timezone back to nothing, i.e. let the operating system decide
    // what to use
    BOOST_TEST_REQUIRE(ml::core::CTimezone::setTimezone(""));
}

BOOST_AUTO_TEST_CASE(testDateWords) {
    // These tests assume they're being run in an English speaking country

    LOG_DEBUG(<< "Checking day of week abbreviations");
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Mon"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Tue"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Wed"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Thu"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Fri"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Sat"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Sun"));

    LOG_DEBUG(<< "Checking full days of week");
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Monday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Tuesday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Wednesday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Thursday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Friday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Saturday"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Sunday"));

    LOG_DEBUG(<< "Checking non-days of week");
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Money"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Tues"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Wedding"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Thug"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Fried"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Satanic"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Sunburn"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Ml"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Dave"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Hello"));

    LOG_DEBUG(<< "Checking month abbreviations");
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Jan"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Feb"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Mar"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Apr"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("May"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Jun"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Jul"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Aug"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Sep"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Oct"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Nov"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("Dec"));

    LOG_DEBUG(<< "Checking full months");
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("January"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("February"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("March"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("April"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("May"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("June"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("July"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("August"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("September"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("October"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("November"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("December"));

    LOG_DEBUG(<< "Checking non-months");
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Jane"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Febrile"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Market"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Apricot"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Maybe"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Junk"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Juliet"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Augment"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Separator"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Octet"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Novel"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Decadent"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Table"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Chair"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("Laptop"));

    LOG_DEBUG(<< "Checking time zones");
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("GMT"));
    BOOST_TEST_REQUIRE(ml::core::CTimeUtils::isDateWord("UTC"));

    LOG_DEBUG(<< "Checking space");
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord(""));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord(" "));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord("\t"));
    BOOST_TEST_REQUIRE(!ml::core::CTimeUtils::isDateWord(" \t"));
}

BOOST_AUTO_TEST_CASE(testDurationToString) {
    ml::core_t::TTime s = 1;
    ml::core_t::TTime m = 60 * s;
    ml::core_t::TTime h = 60 * m;
    ml::core_t::TTime d = 24 * h;
    BOOST_REQUIRE_EQUAL("0s", ml::core::CTimeUtils::durationToString(0));
    BOOST_REQUIRE_EQUAL("1s", ml::core::CTimeUtils::durationToString(s));
    BOOST_REQUIRE_EQUAL("1m", ml::core::CTimeUtils::durationToString(m));
    BOOST_REQUIRE_EQUAL("1h", ml::core::CTimeUtils::durationToString(h));
    BOOST_REQUIRE_EQUAL("1d", ml::core::CTimeUtils::durationToString(d));
    BOOST_REQUIRE_EQUAL("1m1s", ml::core::CTimeUtils::durationToString(m + s));
    BOOST_REQUIRE_EQUAL("1h1m", ml::core::CTimeUtils::durationToString(h + m));
    BOOST_REQUIRE_EQUAL("1h1s", ml::core::CTimeUtils::durationToString(h + s));
    BOOST_REQUIRE_EQUAL("1h1m1s", ml::core::CTimeUtils::durationToString(h + m + s));
    BOOST_REQUIRE_EQUAL("1d1h1m1s", ml::core::CTimeUtils::durationToString(d + h + m + s));
    BOOST_REQUIRE_EQUAL("7d12h", ml::core::CTimeUtils::durationToString(7 * d + 12 * h));
    BOOST_REQUIRE_EQUAL("365d5h48m46s", ml::core::CTimeUtils::durationToString(
                                            365 * d + 5 * h + 48 * m + 46 * s));
}

BOOST_AUTO_TEST_CASE(testTimeDurationStringToSeconds) {
    bool parsedOk{false};
    ml::core_t::TTime durationSeconds{0};
    const ml::core_t::TTime defaultValue{1};

    {
        // All valid and specifying a whole number of seconds
        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("14d", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(1209600, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("14D", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(1209600, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("24h", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(86400, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("24H", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(86400, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("15m", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(900, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("15M", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(900, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("30s", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(30, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("30S", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(30, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2000ms", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(2, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2000MS", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(2, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("3000000micros", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(3, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("3000000MICROS", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(3, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("3000000MiCrOs", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(3, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("4000000000nanos", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(4, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("4000000000NANOS", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(4, durationSeconds);
    }

    {
        // Valid and equating to a fractional number of seconds. We expect the returned value to be rounded
        // down to the nearest whole number of seconds
        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2500ms", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(2, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("3800000micros", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(3, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("4900000000nanos", defaultValue);
        BOOST_TEST_REQUIRE(parsedOk);
        BOOST_REQUIRE_EQUAL(4, durationSeconds);
    }

    {
        // All invalid formats
        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2w", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("14days", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("24hrs", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("15minutes", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("30seconds", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2.5s", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) =
            ml::core::CTimeUtils::timeDurationStringToSeconds("2000millis", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) = ml::core::CTimeUtils::timeDurationStringToSeconds(
            "3000000Microseconds", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);

        std::tie(durationSeconds, parsedOk) = ml::core::CTimeUtils::timeDurationStringToSeconds(
            "4000000000nanoseconds", defaultValue);
        BOOST_TEST_REQUIRE(!parsedOk);
        BOOST_REQUIRE_EQUAL(1, durationSeconds);
    }
}

BOOST_AUTO_TEST_SUITE_END()
